# Clustering Related Posts

## Scenario
- We are implementing a search engine,
- When a user comes in and type in some keywords, how can we find the related posts in our training data?

## Analysis
- Notes on [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning) and [unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning).

- Measuring relateness among posts
    + [Edit distance](https://en.wikipedia.org/wiki/Edit_distance), measures the minimum required operations (insert/replace/delete) on characters to tranfrom one word into the other, e.g. the edit distance of "cat" and "act" is 2, delete "c" then insert "c". The same concept applies to posts, the edit distance among posts can be calculated the minimum required operations on words, instead of characters. Major drawback: Not taking words order into account.

    + [Bag of words](https://en.wikipedia.org/wiki/Bag-of-words_model), example:
    > Two samples:
    > 
    > (1) John likes to watch movies. Mary likes movies too.
    >
    > (2) John also likes to watch football games.
    > 
    > Unique words Occurred:
    > 
    > [
    >     "John",
    >     "likes",
    >     "to",
    >     "watch",
    >     "movies",
    >     "also",
    >     "football",
    >     "games",
    >     "Mary",
    >     "too"
    > ]
    > 
    > Count occurrences accordingly (_vectorization_):
    > 
    > (1) [1, 2, 1, 1, 2, 0, 0, 0, 1, 1]
    >
    > (2) [1, 1, 1, 1, 0, 1, 1, 1, 0, 0]
    
    + Converting raw texts into bag-of-words model using [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) in sklearn:

    ```python
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    print vectorizer
    # output:
    CountVectorizer(analyzer=u'word', binary=False, decode_error=u'strict',
            dtype=<type 'numpy.int64'>, encoding=u'utf-8', input=u'content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), preprocessor=None, stop_words=None,
            strip_accents=None, token_pattern=u'(?u)\\b\\w\\w+\\b',
            tokenizer=None, vocabulary=None)
    ```

    + Feed sample data then start vectorize:

    ```python
    content = ["How to format my hard disk", " Hard disk format problems "]
    X = vectorizer.fit_transform(content)
    print vectorizer.get_feature_names()
    # output: [u'disk', u'format', u'hard', u'how', u'my', u'problems', u'to']
    print X.toarray().T
    # output:
    [[1 1]
     [1 1]
     [1 1]
     [1 0]
     [1 0]
     [0 1]
     [1 0]]
    ```
