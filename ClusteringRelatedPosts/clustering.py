
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

plot(x, y, 'vectors')
plt.savefig('./pics/figure1.png')
plt.clf()


def plot_kmean_iter(i):
    km = KMeans(init='random', n_clusters=numClusters, verbose=1,
                n_init=1, max_iter=i,
                random_state=seed)
    km.fit(features)
    plot(x, y, 'iter %d' % i, km)
    mx, my = sp.meshgrid(sp.arange(0, 1, 0.001), sp.arange(0, 1, 0.001))
    Z = km.predict(sp.vstack((mx.ravel(), my.ravel())).T).reshape(mx.shape)
    plt.imshow(Z, interpolation='nearest',
                 extent=(mx.min(), mx.max(), my.min(), my.max()),
                 cmap=plt.cm.Blues,
                 aspect='auto', origin='lower')
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                  marker='x', linewidth=2, s=100, color='black')
    return km.cluster_centers_

features = sp.vstack((x, y)).T

centers1 = plot_kmean_iter(1)
plt.savefig('./pics/figure2.png')
plt.clf()


def plot_arrow(centers1, centers2):
    for i in range(centers1.shape[0]):
        x1, y1 = centers1[i, :]
        x2, y2 = centers2[i, :]
        plt.arrow(x1, y1, x2 - x1, y2 - y1, length_includes_head=True, head_width=.01, lw=2)

centers2 = plot_kmean_iter(2)
plot_arrow(centers1, centers2)
plt.savefig('./pics/figure3.png')
plt.clf()

centers3 = plot_kmean_iter(3)
plot_arrow(centers2, centers3)
plt.savefig('./pics/figure4.png')
plt.clf()

centers3 = plot_kmean_iter(10)
plt.savefig('./pics/figure5.png')
plt.clf()
