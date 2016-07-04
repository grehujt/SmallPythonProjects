from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
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
print X_train.toarray().T
