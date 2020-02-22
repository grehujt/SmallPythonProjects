# import time
# import asyncio
# 
# async def main():
#     print(f'hi {time.time()}')
#     await asyncio.sleep(1)
#     print(f'world {time.time()}')
# 
# def blocking():
#     time.sleep(.5)
#     print(f'b {time.time()}')
# 
# 
# loop = asyncio.get_event_loop()
# t = loop.create_task(main())
# loop.run_in_executor(None, blocking)
# loop.run_until_complete(t)
# loop.close()
# 
# async def f():
#     try:
#         while True: await asyncio.sleep(0)
#     except asyncio.CancelledError:
#         print('task cancelled.. no')
#         while True: await asyncio.sleep(0)
#     else:
#         return 123
# 
# # a = f()
# # a.send(None)
# # a.send(None)
# # a.throw(asyncio.CancelledError)
# # a.send(None)

import numpy as np
a = np.zeros((2,2))
# print(np.pad(a, 1, 'constant', constant_values=10))

# print(np.diag(np.arange(4)+1, k=-1))
# print(np.unravel_index(99, (6,7,8)))
# print(np.tile(np.array([[0,1],[1,0]]), (4,4)))

color = np.dtype([
    ('r', np.ubyte, (1,)),
    ('g', np.ubyte, (1,)),
    ('b', np.ubyte, (1,))
])
# print(color)

# print(sum(range(5), -1))

# print(np.sum(range(5), -1))

a = np.ones(10)
# print(a**a)

# print(np.array(0) / np.array(0))
# print(np.array(0) // np.array(0))
# print(np.array([np.nan]).astype(int).astype(float))

Z = np.random.uniform(-10,+10,10)
# print(Z)
# print(np.round(Z))
# print(np.copysign(np.ceil(np.abs(Z)), Z))

a = np.random.randint(0, 10, 10)
b = np.random.randint(0, 10, 10)
# print(a)
# print(b)
# print(np.intersect1d(a, b))
#
# print(np.sqrt(-1) == np.emath.sqrt(-1))

yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today = np.datetime64('today', 'D')
tomorrow = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
# print(yesterday, today, tomorrow)
# 
# print(np.arange('2020-01', '2020-02', dtype='datetime64[D]'))

a = np.zeros((5,5), dtype=[('x', np.float64), ('y', np.float64)])
a['x'], a['y'] = np.meshgrid(np.linspace(0,1,5), np.linspace(0,1,5))
# print(a['x'])
# print(a['y'])
# print(a)

# for dt in [np.int8, np.int16, np.int32, np.int64]:
#     print(np.iinfo(dt).min)
#     print(np.iinfo(dt).max)
# for dt in [np.float16, np.float32, np.float64]:
#     print(np.finfo(dt).min)
#     print(np.finfo(dt).max)

a = np.arange(10)
v = np.random.uniform(0, 10)
# print(a, v)
# print(a[np.argmin(np.abs(a-v))])

a = np.zeros(3, [
    ('p', [('x', float), ('y', float)]),
    ('c', [('r', np.uint8), ('g', np.uint8), ('b', np.uint8)])
])
# print(a[0])
# print(a['c']['r'])
# print(a)

a = np.random.random((10, 2))
x, y = np.atleast_2d(a[:, 0], a[:, 1])
# print(x.shape, y.shape, a[:, 0].shape, a[:, 1].shape)


# import dis
# def f():
#     print(((x-x.T)**2+(y-y.T)**2)**.5)
# def f2():
#     print(np.sqrt(np.power(x - x.T, 2) + np.power(y - y.T, 2)))
# dis.dis(f)
# print('asdsa')
# dis.dis(f2)

from io import StringIO
s = StringIO('''1, 2, 3, 4, 5

                6,  ,  , 7, 8

                 ,  , 9,10,11
''')
# print(np.genfromtxt(s, delimiter=','))

a = np.arange(9).reshape((3, 3))
# for i, v in np.ndenumerate(a):
#     print(i, v)
#
# for i in np.ndindex(a.shape):
#     print(i, a[i])

n = 10
p = 3
a = np.zeros((n*n))
# print(np.random.choice(range(n*n), p, replace=False))
# np.put(a, np.random.choice(range(n*n), p, replace=False), p)
# print(a)

a = np.zeros(10)
i = np.array([1,2,1])  ####################
a[i] += 1
# print(a)
a = np.zeros(10)
np.add.at(a, i, 1)
# print(a)

# np.bincount()

a = np.ones((5,5,3))
# print(np.ones((5,5))[:,:,None].shape)

a = np.arange(9).reshape((3,3))
a[[0,1]] = a[[1,0]]
# print(a)

a = np.arange(9).reshape((3,3))
# print(np.roll(a, -1, 1))
# print(np.repeat(a, 2, 0))

# print(len(np.zeros((10,6))))
# print(np.unique([1,2,1]))

def distance(P0, P1, p):
    T = P1 - P0
    L = (T**2).sum(axis=1)
    U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
    U = U.reshape(len(U),1)
    D = P0 + U*T - p
    return np.sqrt((D**2).sum(axis=1))


P0 = np.random.uniform(-10,10,(10,2))
P1 = np.random.uniform(-10,10,(10,2))
p = np.random.uniform(-10,10,(1,2))

# print(distance(P0, P1, p))
# print(np.abs(np.cross(P0-p, P1-p, axis=1) / np.linalg.norm(P0-P1, axis=1)))

a = np.arange(1, 15, dtype=np.int32)
# print(np.lib.stride_tricks.as_strided(a, (11,4), (4,4)))

a = np.random.uniform(0, 1, (5,5))
u, s, v = np.linalg.svd(a)
# print(rank := np.sum(s>1e-6))

a = np.random.randint(1, 10, (20,))
# print(a)
# print(np.bincount(a))
# print(np.argmax(np.bincount(a)))

Z = np.arange(100).reshape(10,10)
C = np.lib.stride_tricks.as_strided(Z, shape=(8, 8, 3, 3), strides=Z.strides + Z.strides)
# print(C)

Z = np.ones((16,16))
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                    np.arange(0, Z.shape[1], k), axis=1)
# print(S)

# import time
# a = np.random.random(1000000)
# t = time.time()
# print(a[np.argsort(a)[-10:]])
# print(time.time()-t)
# t = time.time()
# print(a[np.argpartition(-a, 10)[:10]])####fast
# print(time.time()-t)

# x = np.random.rand(int(5e7))
# from timeit import timeit
# print(timeit('np.power(x,3)', setup='import numpy as np;x = np.random.rand(int(1e7));', number=10))
# print(timeit('x*x*x', setup='import numpy as np;x = np.random.rand(int(1e7));', number=10))
# print(timeit("np.einsum('i,i,i->i',x,x,x)", setup='import numpy as np;x = np.random.rand(int(1e7));', number=10))

# a = np.random.randint(0,2,(3,3))
# b = np.random.randint(0,2,(2,2))
a = np.array([
    [1,0,1],
    [1,1,0],
    [0,1,1]
])
b = np.array([
    [0,1],
    [0,0]
])
# print(a)
# print(b)
c = a[..., np.newaxis, np.newaxis] == b
# print(c)
# print(c.any((3)).all(1))
# print(c.any(1))
# print('-'*10)
# print(c.any((3,1)))
# # print((a[..., np.newaxis, np.newaxis]==b).shape)
# np.any()

# a = np.random.randint(0,2,(3,3,3))
# print(a)
# print(a.any((0)))

a = np.random.randint(0,3,(10,3))
# print(a)
# print('-'*10)
# print(a[np.any(a[:,1:]!=a[:,:-1],1)])

I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
# print(np.unpackbits(I[:, np.newaxis], axis=1))

a = np.random.randint(0,2,(6,2))
# print(a)
# print('-'*10)
# print(np.unique(a, axis=0))

a = np.random.randn(100)
idx = np.random.randint(0,len(a), (1000, 100))
mean = np.mean(a[idx], 1)
print(np.percentile(mean, [2.5, 97.5]))
