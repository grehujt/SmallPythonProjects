
import os
from itertools import count
import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk.stem
from sklearn.cluster import KMeans

stemmer = nltk.stem.SnowballStemmer('english')


class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (stemmer.stem(w) for w in analyzer(doc))


trainDir = './data/20news-bydate-train'
testDir = './data/20news-bydate-test'

groups = set(['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'sci.space'])


def load_20_news_groups(d):
    data = []
    target_names = []
    filenames = []
    for parent, folder, files in os.walk(d):
        target_name = os.path.split(parent)[-1]
        if target_name not in groups:
            continue
        for f in files:
            fullFileName = os.path.join(parent, f)
            data.append(open(fullFileName).read())
            target_names.append(target_name)
            filenames.append(fullFileName)
    return {'data': data, 'target_names': target_names, 'filenames': filenames}

trainData = load_20_news_groups(trainDir)
print len(trainData['filenames'])  # 3529
testData = load_20_news_groups(testDir)
print len(testData['filenames'])  # 4713

vectorizer = StemmedTfidfVectorizer(min_df=10, max_df=0.5, stop_words='english', decode_error='ignore')
trainMat = vectorizer.fit_transform(trainData['data'])
print trainMat.shape

numClusters = 50
km = KMeans(n_clusters=numClusters, n_init=1, verbose=1, random_state=3)
km.fit(trainMat)

newPost = '''
"Disk drive problems.
    Hi, I have a problem with my hard disk. After 1 year it is working only sporadically now.
    I tried to format it, but now it doesn't boot any more.
    Any ideas? Thanks."
'''

newPostVec = vectorizer.transform([newPost])
newPostLabel = km.predict(newPostVec)[0]
print newPostLabel
candidateIndices = (km.labels_ == newPostLabel).nonzero()[0]
print len(candidateIndices)
dists = sp.zeros_like(candidateIndices, dtype=sp.float32)
for i, idx in enumerate(candidateIndices):
    dists[i] = sp.linalg.norm((newPostVec - trainMat[idx, :]).toarray())
sortedIndices = sp.argsort(dists)

print trainData['data'][candidateIndices[sortedIndices[0]]]
print '-' * 30

print trainData['data'][candidateIndices[sortedIndices[20]]]
print '-' * 30

print trainData['data'][candidateIndices[sortedIndices[50]]]
print '-' * 30
