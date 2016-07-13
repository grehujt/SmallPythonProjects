
import os
import scipy as sp
import nltk.stem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.spatial import distance
import matplotlib.pyplot as plt

stemmer = nltk.stem.SnowballStemmer('english')


class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
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
# print trainData['data'][0],'\n' * 5
# print trainData['data'][596],'\n' * 5
# exit(0)
testData = load_20_news_groups(testDir)
print len(testData['filenames'])  # 4713

vectorizer = StemmedTfidfVectorizer(min_df=2, max_df=0.95, stop_words='english', decode_error='ignore')
trainMat = vectorizer.fit_transform(trainData['data'])
print trainMat.shape
testMat = vectorizer.transform(testData['data'])

lda = LatentDirichletAllocation(n_topics=100, random_state=0)
lda.fit(trainMat)
result = sp.sum(lda.transform(testMat) > 0.01, axis=1)
plt.hist(result, sp.arange(1, 10))
plt.xlabel('num of topics')
plt.ylabel('num of posts')
plt.grid()
plt.tight_layout()
plt.savefig('pics/fig6.png')
# exit(0)

# def draw_words_cloud(featureNames, components)

testIndex = 3

newPost = trainData['data'][testIndex]


print 'New post:', '=' * 50
print newPost


def print_top_words(model, feature_names, n):
    topic = model.components_[n, :]
    print "Topic #%d:" % n
    print " ".join([feature_names[i] for i in topic.argsort()[:-30 - 1:-1]])
    print

# feature_names = vectorizer.get_feature_names()
# print_top_words(lda, feature_names, topic)

# exit(0)
# v = newPostTopicVec.T

trainTopicMat = lda.transform(trainMat)

pairwise = distance.squareform(distance.pdist(trainTopicMat))
print pairwise.shape
maximum = pairwise.max()
for i in xrange(pairwise.shape[0]):
    pairwise[i][i] = maximum + 1

print 'closet post:', '*' * 50
print pairwise[testIndex], sp.argmin(pairwise[testIndex]), pairwise[testIndex].min()
print trainData['data'][sp.argmin(pairwise[testIndex])]

# print trainData['data'][sortedIndices[-10]]
# print '*' * 70, sims[sortedIndices[-10]]

# print trainData['data'][sortedIndices[0]]
# print '*' * 70, sims[sortedIndices[0]]
