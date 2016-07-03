
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from itertools import combinations
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

data = np.genfromtxt('seeds_dataset.txt')
features, labels = data[:, 0:7], data[:, -1]

classifier = KNeighborsClassifier(n_neighbors=1)

errs = []
for train, test in KFold(features.shape[0], 5, True):
    classifier.fit(features[train], labels[train])
    pred = classifier.predict(features[test])
    err = (pred == labels[test]).mean()
    errs.append(err)
print '%.3f' % np.mean(errs)

for i1, i2 in combinations(range(features.shape[1]), 2):
    features2 = features[:, (i1, i2)]
    errs = []
    for train, test in KFold(features.shape[0], 5, True):
        classifier.fit(features2[train], labels[train])
        pred = classifier.predict(features2[test])
        errs.append((pred == labels[test]).mean())
    print i1, i2, '%.3f' % np.mean(errs)

featureNames = ['area', 'perimeter', 'compactness', 'length of kernel', 'width of kernel', 'asymmetry coefficient', 'length of kernel groove']


def plot_2d_knn_decision(featureIndices, features, labels, num_neighbors=1):
    '''
    this function is taken from:
    https://github.com/grehujt/BuildingMachineLearningSystemsWithPython/blob/master/ch02/figure4_5_sklearn.py
    '''
    # f1 = features[:, featureIndices[0]]
    # f2 = features[:, featureIndices[1]]
    # x0, x1 = f1.min() * .9, f1.max() * 1.1
    # y0, y1 = f2.min() * .9, f2.max() * 1.1
    f1 = preprocessing.scale(features[:, featureIndices[0]])
    f2 = preprocessing.scale(features[:, featureIndices[1]])
    x0, x1 = f1.min() * 1.1, f1.max() * 1.1
    y0, y1 = f2.min() * 1.1, f2.max() * 1.1

    X = np.linspace(x0, x1, 1000)  # shape: (1000,)
    Y = np.linspace(y0, y1, 1000)  # shape: (1000,)
    X, Y = np.meshgrid(X, Y)  # X.shape: (1000, 1000), Y.shape: (1000, 1000)

    model = KNeighborsClassifier(num_neighbors)
    # model = Pipeline([
    #     ('norm', StandardScaler()),
    #     ('knn', model)
    # ])
    model.fit(np.vstack((f1, f2)).T, labels)
    C = model.predict(np.vstack([X.ravel(), Y.ravel()]).T).reshape(X.shape)
    cmap = ListedColormap([(1., .7, .7), (.7, 1., .7), (.7, .7, 1.)])

    fig, ax = plt.subplots()  # concise way of "fig = plt.figure(); ax = fig.add_subplot(111)"
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel(featureNames[featureIndices[0]])
    ax.set_ylabel(featureNames[featureIndices[1]])
    ax.pcolormesh(X, Y, C, cmap=cmap)

    cmap = ListedColormap([(1., .0, .0), (.1, .6, .1), (.0, .0, 1.)])
    # c=labels, use labels to color data points, mapping to cmap
    ax.scatter(f1, f2, c=labels, cmap=cmap)

    return fig, ax

# fig, ax = plot_2d_knn_decision([3, 6], features, labels, 1)
# fig.savefig('./pics/figure1.png')

# fig, ax = plot_2d_knn_decision([0, 2], features, labels, 1)
# fig.savefig('./pics/figure2.png')

fig, ax = plot_2d_knn_decision([3, 6], features, labels, 1)
fig.savefig('./pics/figure1.1.png')

fig, ax = plot_2d_knn_decision([0, 2], features, labels, 1)
fig.savefig('./pics/figure2.1.png')
