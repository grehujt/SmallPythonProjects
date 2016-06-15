

ZERO_THRES = 1e-14


class Point(object):
    def __init__(self, x=0, y=0):
        super(Point, self).__init__()
        self.x = x
        self.y = y

    def __str__(self):
        return '%f, %f' % (self.x, self.y)

    def __eq__(self, p):
        return abs(self.x - p.x) < ZERO_THRES and abs(self.y - p.y) < ZERO_THRES

    def __add__(self, p):
        return Point(self.x + p.x, self.y + p.y)

    def __sub__(self, p):
        return Point(self.x - p.x, self.y - p.y)

    def __mul__(self, d):
        return Point(self.x * d, self.y * d)


def length_0(a, b):
    return ((a * a) + (b * b))**0.5


def length(v):
    return (v.x * v.x + v.y * v.y)**0.5


def dot_v1(v1, v2):
    return v1.x * v2.x + v1.y * v2.y


def dot_v2(o, p1, p2):
    return (p1.x - o.x) * (p2.x - o.x) + (p1.y - o.y) * (p2.y - o.y)


def cross_v1(v1, v2):
    return v1.x * v2.y - v1.y * v2.x


def cross_v2(o, p1, p2):
    return (p1.x - o.x) * (p2.y - o.y) - (p1.y - o.y) * (p2.x - o.x)


def sign(f):
    return 0 if abs(f) < ZERO_THRES else (1 if f > 0 else -1)


def intersect1D(t, a, b):
    if a > b:
        a, b = b, a
    return t > a and t <= b


def intersect1D_v2(a1, a2, b1, b2):
    if a1 > a2:
        a1, a2 = a2, a1
    if b1 > b2:
        b1, b2 = b2, b1
    return max(a1, b1) <= min(b1, b2)


def dist(p, p1, p2):
    v = p2 - p1
    v1 = p - p1
    return abs(cross_v1(v, v1)) * 1.0 / length(v)


def dist2seg(p, p1, p2):
    v = p2 - p1
    v1 = p - p1
    v2 = p - p2
    if dot_v1(v, v1) <= 0:
        return length(v1)
    if dot_v1(v, v2) >= 0:
        return length(v2)
    return abs(cross_v1(v, v1)) * 1.0 / length(v)


def getPedal(a, p1, p2):
    p1a = Point(a.x - p1.x, a.y - p1.y)
    p1p2 = Point(p2.x - p1.x, p2.y - p1.y)
    ratio = abs(dot_v1(p1a, p1p2)) * 1.0 / (p1p2.x**2 + p1p2.y**2)
    return Point(p1p2.x * ratio + p1.x, p1p2.y * ratio + p1.y)


def getProjection2Seg(p, p1, p2):
    v = p2 - p1
    v1 = p - p1
    v2 = p - p2
    if dot_v1(v, v1) <= 0:
        return p1
    if dot_v1(v, v2) >= 0:
        return p2
    return getPedal(p, p1, p2)


def polarCmp(p0, p1, p2):
    c = sign(cross_v2(p0, p1, p2))
    if c > 0:
        return 1
    if c < 0:
        return -1
    return 1 if length(p1 - p0) > length(p2 - p0) else -1


class Polygon(object):
    def __init__(self):
        super(Polygon, self).__init__()
        self.clear()

    def __str__(self):
        return '%f,%f,%f,%f,%d' % (self.xmin, self.xmax, self.ymin, self.ymax, len(self.pts))

    def clear(self):
        self.xmin = 0
        self.xmax = 0
        self.ymin = 0
        self.ymax = 0
        self.size = 0
        self.kernel = Point(0, 0)
        self.pts = []

    def load(self, polyId, vs):
        self.clear()
        if not vs:
            return
        tmp = vs
        if len(tmp) < 3:
            print 'FATAL ERROR: len(tmp)<3'
            exit(-1)
        self.polyId = polyId
        self.xmin = float(tmp[0])
        self.xmax = float(tmp[0])
        self.ymin = float(tmp[1])
        self.ymax = float(tmp[1])

        for i in range(0, len(tmp), 2):
            p = Point(float(tmp[i]), float(tmp[i + 1]))
            if p.x < self.xmin:
                self.xmin = p.x
            if p.x > self.xmax:
                self.xmax = p.x
            if p.y < self.ymin:
                self.ymin = p.y
            if p.y > self.ymax:
                self.ymax = p.y
            self.kernel.x += p.x  # must be convex
            self.kernel.y += p.y
            self.pts.append(p)
        self.size = len(self.pts)
        self.kernel.x /= self.size
        self.kernel.y /= self.size
        self.pts.sort(cmp=lambda p1, p2: polarCmp(
            self.kernel, p1, p2))  # anti-clockwise sort

    def isWithin_v1(self, p):  # v1, accelerated version for simple polygons
        isInside = False
        if intersect1D(p.x, self.xmin, self.xmax) and intersect1D(p.y, self.ymin, self.ymax):
            size = self.size
            j = size - 1
            for i in range(size):
                p1 = self.pts[i]
                p2 = self.pts[j]
                if p == p1 or p == p2:
                    return False
                if sign(cross_v2(p, p1, p2)) == 0 and dot_v2(p, p1, p2) < 0:
                    return False
                if intersect1D(p.y, p1.y, p2.y) and (p1.x <= p.x or p2.x <= p.x):
                    isInside ^= (p.x > p1.x + (p2.x - p1.x) *
                                 (p.y - p1.y) / (p2.y - p1.y))
                j = i
        return isInside

    def binary_search(self, p):
        size = self.size
        i = 1
        j = size - 1
        ori = self.pts[0]
        e = p - ori
        e1 = self.pts[i] - ori
        e2 = self.pts[j] - ori
        if sign(cross_v1(e1, e)) * sign(cross_v1(e2, e)) >= 0:
            if sign(dot_v1(e2, e)) >= 0:
                return j, 0
            elif sign(dot_v1(e1, e)) >= 0:
                return 0, i
        while i + 1 < j:
            mid = (i + j) / 2
            e_mid = self.pts[mid] - ori
            if sign(cross_v1(e_mid, e)) < 0:
                i = mid
            else:
                j = mid
        return i, j

    def isWithin_v2(self, p):  # v2, accelerated version for convex ONLY
        if intersect1D(p.x, self.xmin, self.xmax) and intersect1D(p.y, self.ymin, self.ymax):
            i, j = self.binary_search(p)
            p1 = self.pts[i]
            p2 = self.pts[j]
            return True if sign(cross_v2(p1, p2, p)) <= 0 else False
        return False

    def isWithin_v3(self, p):  # v3, convex ONLY, also return nearest edge
        isInside = False
        d = 100000000
        ni, nj = Point(-1, -1), Point(-1, -1)
        size = self.size
        j = size - 1
        for i in range(size):
            p1 = self.pts[i]
            p2 = self.pts[j]
            tmp = dist2seg(p, p1, p2)
            if d > tmp:
                ni, nj = p1, p2
                d = tmp
            if intersect1D(p.y, p1.y, p2.y) and (p1.x <= p.x or p2.x <= p.x):
                isInside ^= (p.x > p1.x + (p2.x - p1.x) *
                             (p.y - p1.y) / (p2.y - p1.y))
            j = i
        return isInside, d, ni, nj
