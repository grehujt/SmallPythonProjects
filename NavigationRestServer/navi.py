

import os
import cPickle as pickle
import ujson as json
from itertools import groupby, combinations

import requests
from scipy import spatial
import graph_tool as gt
from graph_tool.topology import shortest_path
# from graph_tool.draw import graph_draw

import poly_math as PM


class NavigaorMgr:
    def __init__(self, conf):
        self._clear()
        self._load(conf)

    def navigate(self, jsonStr):
        # try:
        doc = json.loads(jsonStr)
        site = doc['site']
        srcLoc = (doc['from']['area'], doc['from']['x'], doc['from']['y'])
        dstLoc = (doc['to']['area'], doc['to']['x'], doc['to']['y'])
        nav = self.site_navigator[site]
        return nav.navigate(srcLoc, dstLoc)
        # except Exception, e:
        #     raise e

    def _clear(self):
        self.conf = {}
        self.site_navigator = {}

    def _load(self, conf):
        self.conf = json.load(open(conf))
        for site, doc in self.conf.iteritems():
            nav = SiteNavigator(site, doc['floorHeight'], doc['edgeUrl'], doc[
                                'connectorUrl'], doc['regionUrl'])
            self.site_navigator[site] = nav


class SiteNavigator:
    def __init__(self, site, floorHeight, edgesUrl, connectorUrl, regionsUrl):
        self.site = site
        self.floorHeight = floorHeight
        self.allLocs, self.allEdges, self.areaId_pts = self._initEdges(
            edgesUrl)
        self._initConnectors(connectorUrl, self.allLocs, self.areaId_pts)
        self.areaId_polys = self._initPolys(regionsUrl)
        self.areaId_kdTree = self._initKdTree(self.areaId_pts)
        self.loc_polyIds, self.polyId_locs = self._genEdgesForPtsFromSameRegion(
            self.areaId_pts, self.areaId_polys, self.allEdges)
        self.graph, self.locId_loc, self.loc_locId = self._initGraph(
            self.allLocs, self.allEdges)

    def _initEdges(self, url):
        edgePkl = 'naviData/edges.%s.pkl' % self.site
        if os.path.exists(edgePkl):
            edgeItems = pickle.load(open(edgePkl))
        else:
            edgeItems = json.loads(requests.get(url, verify=False).content)
            with open(edgePkl, 'wb') as fout:
                pickle.dump(edgeItems, fout)
        assert edgeItems
        print len(edgeItems)

        allLocs = set()
        allEdges = {}
        areaId_pts = {}
        for e in edgeItems:
            fromPt, toPt = json.loads(e['fromPt']), json.loads(e['toPt'])
            fromArea, toArea = e['fromArea'], e['toArea']
            srcLoc = (fromArea, fromPt[0], fromPt[1])
            dstLoc = (toArea, toPt[0], toPt[1])

            if fromArea not in areaId_pts:
                areaId_pts[fromArea] = set()
            if toArea not in areaId_pts:
                areaId_pts[toArea] = set()
            areaId_pts[fromArea].add(tuple(fromPt))
            areaId_pts[toArea].add(tuple(toPt))
            if (srcLoc, dstLoc) not in allEdges and (dstLoc, srcLoc) not in allEdges:
                allEdges[(srcLoc, dstLoc)] = PM.length_0(fromPt[0] - toPt[0],
                                                         fromPt[1] - toPt[1]) + abs(fromArea - toArea) * self.floorHeight
            allLocs.add(srcLoc)
            allLocs.add(dstLoc)
        return allLocs, allEdges, areaId_pts

    def _initConnectors(self, url, allLocs, areaId_pts):
        connectorPkl = 'naviData/connectors.%s.pkl' % self.site
        if os.path.exists(connectorPkl):
            connectorItems = pickle.load(open(connectorPkl))
        else:
            connectorItems = json.loads(
                requests.get(url, verify=False).content)
            with open(connectorPkl, 'wb') as fout:
                pickle.dump(connectorItems, fout)
        assert connectorItems
        print len(connectorItems)
        for doc in connectorItems:
            areaId = doc['areaId']
            if areaId not in areaId_pts:
                areaId_pts[areaId] = set()
            polyPts = json.loads(doc['pts'])
            assert len(polyPts) % 2 == 0
            for i in xrange(0, len(polyPts), 2):
                loc = (areaId, polyPts[i], polyPts[i + 1])
                allLocs.add(loc)
                areaId_pts[areaId].add((polyPts[i], polyPts[i + 1]))

    def _initPolys(self, url):
        regionPkl = 'naviData/regions.%s.pkl' % self.site
        if os.path.exists(regionPkl):
            polyItems = pickle.load(open(regionPkl))
        else:
            polyItems = json.loads(requests.get(url, verify=False).content)
            with open(regionPkl, 'wb') as fout:
                pickle.dump(polyItems, fout)
        assert polyItems
        print len(polyItems)

        areaId_polys = {}
        for d in polyItems:
            polyId = d['_id']
            poly = PM.Polygon()
            poly.load(polyId, json.loads(d['vertex']))
            areaId = d['areaId']
            if areaId not in areaId_polys:
                areaId_polys[areaId] = []
            areaId_polys[areaId].append(poly)
        return areaId_polys

    def _initKdTree(self, areaId_pts):
        return dict((areaId, spatial.cKDTree(list(pts))) for areaId, pts in areaId_pts.iteritems())

    def _genEdgesForPtsFromSameRegion(self, areaId_pts, areaId_polys, allEdges):
        areaId__polyId_pts = {}
        for areaId, polys in areaId_polys.iteritems():
            if areaId not in areaId_pts:
                continue
            pts = areaId_pts[areaId]
            pts2 = [PM.Point(x, y) for x, y in pts]
            polyId_pts = {}
            for poly in polys:
                polyId_pts[poly.polyId] = [(p.x, p.y)
                                           for p in pts2 if poly.isWithin_v2(p)]
            areaId__polyId_pts[areaId] = polyId_pts

        loc_polyIds, polyId_locs = {}, {}
        for areaId, polyId_pts in areaId__polyId_pts.iteritems():
            for polyId, pts in polyId_pts.iteritems():
                locs = []
                for p1, p2 in combinations(pts, 2):
                    srcLoc = (areaId, p1[0], p1[1])
                    dstLoc = (areaId, p2[0], p2[1])
                    if srcLoc not in loc_polyIds:
                        loc_polyIds[srcLoc] = []
                    if dstLoc not in loc_polyIds:
                        loc_polyIds[dstLoc] = []
                    loc_polyIds[srcLoc].append(polyId)
                    loc_polyIds[dstLoc].append(polyId)
                    locs.append(srcLoc)
                    locs.append(dstLoc)
                    if (srcLoc, dstLoc) not in allEdges and (dstLoc, srcLoc) not in allEdges:
                        allEdges[(srcLoc, dstLoc)] = PM.length_0(
                            p1[0] - p2[0], p1[1] - p2[1])
                polyId_locs[polyId] = locs
        return loc_polyIds, polyId_locs

    def _initGraph(self, allLocs, allEdges):
        allLocs = list(allLocs)
        locId_loc = dict(zip(xrange(len(allLocs)), allLocs))
        loc_locId = dict(zip(allLocs, xrange(len(allLocs))))

        graph = gt.Graph(directed=False)
        graph.add_vertex(len(allLocs))
        vprop_vint = graph.new_vertex_property("vector<int>")
        for v, loc in locId_loc.iteritems():
            vprop_vint[v] = loc
        eprop_double = graph.new_edge_property("double")
        for edge, w in allEdges.iteritems():
            e = graph.add_edge(loc_locId[edge[0]], loc_locId[edge[1]])
            eprop_double[e] = w
        graph.vp['loc'] = vprop_vint
        graph.ep['weight'] = eprop_double
        # graph_draw(graph, output='%s.png' %
        #            self.site, output_size=(1000, 1000))
        return graph, locId_loc, loc_locId

    def _get_polygon_ids(self, loc):
        if loc in self.loc_locId:
            return self.loc_polyIds[loc]
        polys = self.areaId_polys[loc[0]]
        p = PM.Point(loc[1], loc[2])
        return [poly.polyId for poly in polys if poly.isWithin_v2(p)]

    def _from_same_region(self, loc1, loc2):
        pids1 = self._get_polygon_ids(loc1)
        pids2 = self._get_polygon_ids(loc2)
        for pid1 in pids1:
            for pid2 in pids2:
                if pid1 == pid2:
                    return True
        return False

    def get_nearest_node(self, loc):
        if loc in self.loc_locId:
            return loc
        areaId = loc[0]
        p = (loc[1], loc[2])
        kdt = self.areaId_kdTree[areaId]
        ds, indices = kdt.query(p, 10)
        for i in indices:
            tmp = kdt.data[i]
            loc0 = (areaId, int(tmp[0]), int(tmp[1]))
            if self._from_same_region(loc, loc0):
                return loc0
        return None

    def navigate(self, srcLoc, dstLoc):
        src = self.loc_locId[self.get_nearest_node(srcLoc)]
        dst = self.loc_locId[self.get_nearest_node(dstLoc)]
        if src is None or dst is None:
            return -1, []
        graph = self.graph
        vl, el = shortest_path(graph, graph.vertex(src),
                               graph.vertex(dst), graph.ep['weight'])
        start, end = 0, len(vl)
        i = 0
        while i < len(vl):
            loc = graph.vp['loc'][vl[i]]
            i += 1
            if srcLoc[0] == loc[0] and self._from_same_region(srcLoc, loc):
                start += 1
            else:
                break
        i = len(vl) - 1
        while i > 0:
            loc = graph.vp['loc'][vl[i]]
            i -= 1
            if dstLoc[0] == loc[0] and self._from_same_region(dstLoc, loc):
                end -= 1
            else:
                break
        paths = [srcLoc]
        paths.extend(graph.vp['loc'][i] for i in vl[start:end])
        paths.append(dstLoc)
        startLoc = graph.vp['loc'][vl[start]]
        endLoc = graph.vp['loc'][vl[end - 1]]
        dists = PM.length_0(srcLoc[1] - startLoc[1], srcLoc[2] - startLoc[2])
        dists += sum(graph.ep['weight'][e] for e in el[start:end])
        dists += PM.length_0(dstLoc[1] - endLoc[1], dstLoc[2] - endLoc[2])
        resultPts = []
        for k, grp in groupby(paths, key=lambda x: x[0]):
            tmp = {'area': k, 'pts': []}
            for item in grp:
                # tmp['pts'].extend((item[1], item[2]))
                tmp['pts'].append((item[1], item[2]))
            resultPts.append(tmp)
        return dists, json.dumps(resultPts)


class NaviWithLocResource:
    def on_get(self, req, resp):
        try:
            resp.content_type = 'application/json; charset=utf-8'
            srcLoc = tuple(int(i) for i in req.get_param('src').split('-'))
            dstLoc = tuple(int(i) for i in req.get_param('dst').split('-'))
            site = req.get_param('site')
            tmp = {'site': site,
                   'from': {'area': srcLoc[0], 'x': srcLoc[1], 'y': srcLoc[2]},
                   'to': {'area': dstLoc[0], 'x': dstLoc[1], 'y': dstLoc[2]}
                   }
            resp.body = navMgr.navigate(json.dumps(tmp))[1]
        except Exception:
            # print "Exception in user code:"
            # print '-'*60
            # traceback.print_exc(file=sys.stdout)
            # print '-'*60
            resp.status = falcon.HTTP_404
            resp.body = "{}"

if __name__ == '__main__':
    import sys
    import falcon
    import bjoern
    navMgr = NavigaorMgr('nav.json')
    port = int(sys.argv[1])
    print 'navi server start at port', port
    app = falcon.API()
    app.add_route('/test_loc', NaviWithLocResource())  # src,dst
    bjoern.run(app, '0.0.0.0', port, reuse_port=True)
