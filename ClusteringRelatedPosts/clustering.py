
import scipy as sp
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

seed = 2
sp.random.seed(seed)  # to reproduce the data later on

numClusters = 3

xw1 = norm(loc=.3, scale=.15).rvs(20)
yw1 = norm(loc=.3, scale=.15).rvs(20)

xw2 = norm(loc=.7, scale=.15).rvs(20)
yw2 = norm(loc=.7, scale=.15).rvs(20)

xw3 = norm(loc=.2, scale=.15).rvs(20)
yw3 = norm(loc=.8, scale=.15).rvs(20)

x = sp.append(sp.append(xw1, xw2), xw3)
y = sp.append(sp.append(yw1, yw2), yw3)


def plot(x, y, title, km=None):
    if km:
        plt.scatter(x, y, s=50, c=km.predict(sp.vstack((x, y)).T))
    else:
        plt.scatter(x, y, s=50)
    plt.title(title)
    plt.autoscale(tight=True)
    plt.grid()
    plt.xlim(0, 1)
    plt.ylim(0, 1)

i = 1
plot(x, y, 'vectors')
plt.savefig('./pics/figure1.png')
plt.clf()

i += 1
features = sp.vstack((x, y)).T
km = KMeans(init='random', n_clusters=numClusters, verbose=1,
            n_init=1, max_iter=1,
            random_state=seed)
km.fit(features)
plot(x, y, 'iter 1', km)
mx, my = sp.meshgrid(sp.arange(0, 1, 0.001), sp.arange(0, 1, 0.001))
Z = km.predict(sp.vstack((mx.ravel(), my.ravel())).T).reshape(mx.shape)
plt.imshow(Z, interpolation='nearest',
             extent=(mx.min(), mx.max(), my.min(), my.max()),
             cmap=plt.cm.Blues,
             aspect='auto', origin='lower')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
              marker='x', linewidth=2, s=100, color='black')
plt.savefig('./pics/figure2.png')
plt.clf()
