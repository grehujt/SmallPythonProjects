
from gensim import corpora, models, matutils
import matplotlib.pyplot as plt

# https://radimrehurek.com/gensim/corpora/bleicorpus.html
corpus = corpora.BleiCorpus('./data/ap.dat', './data/vocab.txt')
print type(corpus)
print dir(corpus)
# https://radimrehurek.com/gensim/models/ldamodel.html
model = models.ldamodel.LdaModel(corpus, num_topics=100, id2word=corpus.id2word)

# doc = corpus.docbyoffset(0)
# topics = model[doc]
# numTopics = [len(model[d]) for d in corpus]
# hist1= plt.hist(numTopics, alpha=0.5, bins=100, label='alpha=1/topicCount')
# plt.xlabel('num of topics')
# plt.ylabel('num of docs')
# plt.savefig('pics/pic0.png')
# exit(0)

# model = models.ldamodel.LdaModel(corpus, num_topics=100, id2word=corpus.id2word, alpha=1)
# topics = model[doc]
# numTopics = [len(model[d]) for d in corpus]
# hist2 = plt.hist(numTopics, alpha=0.5, bins=100, label='alpha=1')
# plt.legend()
# plt.xlabel('num of topics')
# plt.ylabel('num of docs')
# plt.savefig('pics/pic1.png')


# find most discussed topic
topics = matutils.corpus2dense(model[corpus], num_terms=model.num_topics)
weight = topics.sum(1)
tIdices = weight.argsort()

h = '''| topic id        | words           |
| :-------------: |:-------------:|
'''
print h
for i in xrange(-1, -11, -1):
    words = model.show_topic(tIdices[i])
    print '|', tIdices[i], '|', ' '.join(s[0] for s in sorted(words, key=lambda d: d[1], reverse=True)), '|'

from scipy.spatial import distance
# calculate pairwise distance between topic vectors
pdists = distance.squareform(distance.pdist(topics))
# excluding self
largest = pdists.max()
for i in xrange(pdists.shape[0]):
    pdists[i][i] = largest + 1

import re
rawtext = re.findall(r'(?s)<TEXT>(.*?)</TEXT>', open('./data/ap.txt').read())
testDocIdx = 1
mostSimDocIdx = pdists[testDocIdx].argmin()
print rawtext[testDocIdx]
print
print
print
print rawtext[mostSimDocIdx]


corpora.MmCorpus.serialize('./data/ap.mm', corpus)
mm = corpora.mmcorpus.MmCorpus('./data/ap.mm')
hdp = models.hdpmodel.HdpModel(corpus, corpus.id2word)
print type(hdp)
print dir(hdp)

