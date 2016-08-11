# Detect floor based on WiFi signals using SVM

## Scenario

Given enough wifi signals (vectors) from multi-floor of a mall, implement an algorithm to determine floor when a new wifi signals (vectors) comes.

## input format

- file structure
    + traning data: ./data/testMall/\<floorId\>/train_ref.txt
    + test data: ./data/testMall/\<floorId\>/test_tar.txt
- format
    + one survey point per line, in each line:
        x,y ApMac:rssi ApMac:rssi ..

## key points

- numpy
- sklearn
    + pipe
    + preprocess
    + svm
    + naive bayes

## Solution

```python

import os
import time
import re
import glob
from itertools import count, izip

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

testFolder = r'./data/testMall'


def parseContent(mac_id, content):
    lines = [l.strip() for l in content.splitlines() if ',' in l]
    matrix = np.zeros((len(lines), len(mac_id)), dtype=np.float32)
    ptMat = np.zeros((len(lines), 2), dtype=np.float32)
    for i, line in enumerate(lines):
        items = line.split(' ')
        loc = [float(j) for j in items[0].split(',')]
        try:
            for item in items[1:]:
                tmp = re.split(r':|,', item)
                if tmp[0] not in mac_id:
                    continue
                matrix[i, mac_id[tmp[0]]] = float(tmp[1])
        except Exception, e:
            print 'err'
            print line
            raise e
        ptMat[i, 0] = loc[0]
        ptMat[i, 1] = loc[1]
    matrix = np.power(2, matrix / 10.0)
    matrix[matrix >= 1] = 0
    return ptMat, matrix


def load(pattern):
    macs = set()
    fn_content = {}
    floorId_intAreaId = {}
    fn_id = {}
    for fn in glob.glob(pattern):
        content = open(fn).read()
        fn_content[fn] = content
        floorId = fn.split(os.path.sep)[-2]
        if floorId not in floorId_intAreaId:
            floorId_intAreaId[floorId] = len(floorId_intAreaId)
        fn_id[fn] = floorId_intAreaId[floorId]
        tmp = set(s.lower() for s in re.findall(r'(\w{9,12}):', content))
        if fn.endswith('ref'):
            macs = macs | tmp
        print fn, len(tmp), 'macs'
    print 'total %d macs found' % len(macs)
    mac_id = dict(izip(macs, count()))
    id_mac = dict(izip(count(), macs))
    return macs, mac_id, id_mac, fn_id, fn_content


def transform(mac_id, fn_id, fn_content):
    labelRefs, featureRefs = None, None
    labelTars, featureTars = None, None
    for fn, content in fn_content.iteritems():
        ptMat, matrix = parseContent(mac_id, content)
        labels = np.ones((matrix.shape[0], 1), dtype=np.int32) * fn_id[fn]
        if fn.endswith('ref'):
            if featureRefs is None:
                featureRefs = matrix
                labelRefs = labels
            else:
                featureRefs = np.vstack((featureRefs, matrix))
                labelRefs = np.vstack((labelRefs, labels))
        elif fn.endswith('tar'):
            if featureTars is None:
                featureTars = matrix
                labelTars = labels
            else:
                featureTars = np.vstack((featureTars, matrix))
                labelTars = np.vstack((labelTars, labels))
        else:
            print 'err fn', fn
            exit(0)
    return labelRefs, featureRefs, labelTars, featureTars

start = time.time()
macs, mac_id, id_mac, fn_id, fn_content = load('%s/*/p_all*' % testFolder)
labelRefs, featureRefs, labelTars, featureTars = transform(mac_id, fn_id, fn_content)
print 'loaded in', time.time() - start, 's'

start = time.time()

model = Pipeline([
    ('norm', MinMaxScaler()),
    ('clf', svm.LinearSVC())
])

# clf = GaussianNB()
# clf = MultinomialNB()
# clf = BernoulliNB()
# clf = svm.LinearSVC()

model.fit(featureRefs, labelRefs.ravel())
# p = model.predict(featureTars)
end = time.time()
print model.score(featureTars, labelTars.ravel())
end2 = time.time()

print featureRefs.shape
print featureTars.shape
print end-start, end2-end, (end2-end) / featureTars.shape[0]

```
