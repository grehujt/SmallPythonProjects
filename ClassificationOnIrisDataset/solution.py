
# coding: utf-8

from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cross_validation import KFold

data = load_iris()

for k, v in data.items():
    print k
    print v, '\n'

featureNames = data['feature_names']
features = data['data']
targetNames = data['target_names']
targets = data['target']

plt.figure(figsize=(15, 10))
styles = {0: 'r>', 1: 'go', 2: 'bx'}
for f1, f2 in combinations(range(len(featureNames)), 2):
    plt.subplot(230+f2 if f1==0 else 231+f1+f2)
    plt.grid()
    plt.xlabel(featureNames[f1])
    plt.ylabel(featureNames[f2])
    for t in range(len(targetNames)):
        plt.scatter(features[targets==t, f1], features[targets==t, f2], marker=styles[t][1], c=styles[t][0])
# plt.show()

labels = targetNames[targets]
plen = features[:, 2]
is_setosa = (labels == 'setosa')
print plen[is_setosa].max()
print plen[~is_setosa].min()


def is_setosa_test(features):
    return features[2] < 2.5

x0 = features[:, 2].min() * .8
x1 = features[:, 2].max() * 1.2
y0 = features[:, 3].min() * .8
y1 = features[:, 3].max() * 1.2
plt.figure()
plt.grid()
plt.xlabel(featureNames[2])
plt.ylabel(featureNames[3])
plt.fill_between([x0, 2.5], [y0, y0], [y1, y1], color=(1, .9, .9))
plt.fill_between([2.5, x1], [y0, y0], [y1, y1], color=(.9, .9, 1))
plt.plot([2.5, 2.5], [y0, y1], 'k--', lw=3)
for t in range(len(targetNames)):
    plt.scatter(features[targets==t, 2], features[targets==t, 3], marker=styles[t][1], c=styles[t][0])
plt.xlim(x0, x1)
plt.ylim(y0, y1)
# plt.show()

features = features[~is_setosa]
labels = labels[~is_setosa]
is_virginica = (labels == 'virginica')


def fit_model(features, labels):
    bestThres = 0
    bestAcc = -1
    bestF = 0
    rev = False
    for f in range(features.shape[1]):
        for t in features[:, f]:
            pred = (features[:, f] > t)
            acc = (pred == labels).mean()
            if acc > bestAcc or 1 - acc > bestAcc:
                bestThres = t
                bestAcc = max(acc, 1 - acc)
                bestF = f
                rev = bestAcc == 1 - acc
    return bestThres, bestF, rev

model = fit_model(features, is_virginica)
print model[0], model[1], featureNames[model[2]]

x0 = features[:, 2].min() * .8
x1 = features[:, 2].max() * 1.2
y0 = features[:, 3].min() * .8
y1 = features[:, 3].max() * 1.2
targets = targets[~is_setosa]
plt.figure()
plt.grid()
plt.xlabel(featureNames[2])
plt.ylabel(featureNames[3])
plt.fill_between([x0, x1], [1.6, 1.6], [y0, y0], color=(1, .9, .9))
plt.fill_between([x0, x1], [1.6, 1.6], [y1, y1], color=(.9, .9, 1))
plt.plot([x0, x1], [1.6, 1.6], 'k--', lw=3)
for t in range(len(targetNames)):
    plt.scatter(features[targets==t, 2], features[targets==t, 3], marker=styles[t][1], c=styles[t][0])
plt.xlim(x0, x1)
plt.ylim(y0, y1)
# plt.show()


def predict(model, features):
    t, f, rev = model
    return features[:, f] > t if not rev else features[:, f] <= t


def accuracy(features, labels, model):
    return (predict(model, features) == labels).mean()

for train, test in KFold(labels.shape[0], 5, True):
    model = fit_model(features[train], is_virginica[train])
    print 'train acc:', accuracy(features[train], is_virginica[train], model),
    print 'test acc:', accuracy(features[test], is_virginica[test], model)
