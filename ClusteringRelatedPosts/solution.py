
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer
import nltk.stem

stemmer = nltk.stem.SnowballStemmer('english')


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: (stemmer.stem(w) for w in analyzer(doc))

# vectorizer = CountVectorizer()
# vectorizer = CountVectorizer(stop_words='english')
vectorizer = StemmedCountVectorizer(stop_words='english')
print vectorizer

content = ["How to format my hard disk", " Hard disk format problems "]
X = vectorizer.fit_transform(content)
print vectorizer.get_feature_names()
print X.toarray().T

corpus = [
    "This is a toy post about machine learning. Actually, it contains not much interesting stuff.",
    "Imaging databases can get huge.",
    "Most imaging databases save images permanently.",
    "Imaging databases store images.",
    "Imaging databases store images. Imaging databases store images. Imaging databases store images."
]
X_train = vectorizer.fit_transform(corpus)
print vectorizer.get_feature_names()
print X_train.shape, X_train.toarray().T

newPost = 'imaging databases'
newVec = vectorizer.transform([newPost])
print newVec


def dist(v1, v2):
    # return euclidean distance
    return sp.linalg.norm((v1 - v2).toarray())

minI, minDist = 0, 1e10
for i in range(X_train.shape[0]):
    d = dist(X_train[i, :], newVec)
    print i, d
    if d < minDist:
        minDist = d
        minI = i
print 'most related', minI, minDist


def dist_norm(v1, v2):
    v1_normed = v1 / sp.linalg.norm(v1.toarray())
    v2_normed = v2 / sp.linalg.norm(v2.toarray())
    return sp.linalg.norm((v1_normed - v2_normed).toarray())

minI, minDist = 0, 1e10
for i in range(X_train.shape[0]):
    d = dist_norm(X_train[i, :], newVec)
    print i, d
    if d < minDist:
        minDist = d
        minI = i
print 'most related', minI, minDist

print X_train[1,:]
print
print X_train[3,:]


# stemmer = nltk.stem.SnowballStemmer('english')
print stemmer.stem("image")  # imag
print stemmer.stem("images")  # imag
print stemmer.stem("imaging")  # imag
print stemmer.stem("imagination")  # imagin
