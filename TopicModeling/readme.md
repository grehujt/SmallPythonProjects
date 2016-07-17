＃ Topic Modeling

## Scenario

We have 2246 documents from the Associated Press, and we are going to investigate the topics of them and group similar documents to related topics.

[Click here to download the dataset.](http://www.cs.princeton.edu/~blei/lda-c/ap.tgz)

## Anaysis

- Unlike clustering, one document can talk about more than 1 topics, we need a way to infer the topics from corpus and group them.

- Latent Dirichlet Allocation
    + [wiki](http://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)
    + [paper](http://www.cs.princeton.edu/~blei/papers/BleiNgJordan2003.pdf)
    + Simply put, LDA is a probabilistic model which assigns different weights on words from different topics and allows you to analyze of corpus, and extract the topics to form its documents. The input of the model is a list of documents, and the output is a list of topics with related weighted words.

- Sklearn has implementation of LDA, but the interface is not as friendly as [gensim](https://radimrehurek.com/gensim/)'s, so I am going to use gensim here. Once finished downloading & extracting the zip, we will see thress files:
    + ap.txt, which is the raw text of 2246 documents;
    + ap.dat, BleiC format file, which is the preprocessed version of the documents in ap.txt, contains 2246 lines and each line represents one documents in the following format: 
    
    > docId termId:termCount termId:termCount ...
    
    + vocab.txt, containing all distinct words occurred in the 2246 documents.

- We can import the dataset and feed it into our lda model using gensim:

    (Noted that we set num_topics=100 here)

    ```python
    from gensim import corpora, models
    corpus = corpora.BleiCorpus('./data/ap.dat', './data/vocab.txt')
    print type(corpus)
    print dir(corpus)
    model = models.ldamodel.LdaModel(corpus, num_topics=100, id2word=corpus.id2word)
    # output:
    <class 'gensim.corpora.bleicorpus.BleiCorpus'>
    ['__class__', '__delattr__', '__dict__', '__doc__', '__format__', '__getattribute__', '__getitem__', '__hash__', '__init__', '__iter__', '__len__', '__module__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_adapt_by_suffix', '_load_specials', '_save_specials', '_smart_save', 'docbyoffset', 'fname', 'id2word', 'index', 'length', 'line2doc', 'load', 'save', 'save_corpus', 'serialize']
    ```

    More on the constructor can be found [here](# https://radimrehurek.com/gensim/corpora/bleicorpus.html).

- After feeding our model, we can do some exploring:

    ```python
    doc = corpus.docbyoffset(0)  # grap first doc in the corpus
    topics = model[doc]  # grop the topics it talked about
    print topics
    # [(2, 0.025557527971728191),
    #  (13, 0.019739219934498245),
    #  (16, 0.53534741785582873),
    #  (32, 0.079707457065004594),
    #  (34, 0.016242485776949589),
    #  (38, 0.011064365938014683),
    #  (49, 0.11325518263205145),
    #  (51, 0.012685628601841499),
    #  (54, 0.011589007379718155),
    #  (66, 0.020586404205020049),
    #  (87, 0.035147854577898527),
    #  (89, 0.018680487534867025),
    #  (94, 0.070224719493546028)]

    # plot the hist of topic count in the corpus
    numTopics = [len(model[d]) for d in corpus]
    hist1= plt.hist(numTopics, alpha=0.5, bins=100, label='alpha=1/topicCount', rwidth=0.5)
    plt.grid()
    plt.xlabel('num of topics')
    plt.ylabel('num of docs')
    plt.savefig('pics/pic0.png')
    ```

    ![png](pics/pic0.png)

- One of the most important parameter in the LDA model contructor is alpha. By default, gensim will set alpha to 1/num_topics. The smaller alpha is, the fewer topics each document will be expected to discuss, if we change alpha to 1:

    ```python
    model = models.ldamodel.LdaModel(corpus, num_topics=100, id2word=corpus.id2word, alpha=1)
    topics = model[doc]
    numTopics = [len(model[d]) for d in corpus]
    hist2 = plt.hist(numTopics, alpha=0.5, bins=100, label='alpha=1')
    plt.legend()
    plt.xlabel('num of topics')
    plt.ylabel('num of docs')
    plt.savefig('pics/pic1.png')
    ```

    ![png](pics/pic1.png)

- Extract top 10 hottest topics:

    ```python
    topics = matutils.corpus2dense(model[corpus], num_terms=model.num_topics)
    weight = topics.sum(1)
    tIdices = weight.argsort()

    h = '''| topic id        | words           |
    | :-------------: |:-------------:|'''
    print h
    for i in xrange(-1, -11, -1):
        words = model.show_topic(tIdices[i])
        print '|', tIdices[i], '|', ' '.join(s[0] for s in sorted(words, key=lambda d: d[1], reverse=True)), '|'
    ```

    | topic id        | words           |
    | :-------------: |:-------------:|
    | 36 | bush i new dukakis trade primary states president year campaign |
    | 13 | united billion government states new economic officials world first people |
    | 45 | government united president year i states new military two congress |
    | 63 | police years two ruby school day i million year porter |
    | 30 | i convention president people soviet black new years government research |
    | 73 | percent year billion last million new police years tax spending |
    | 32 | soviet dinner reagan bentsen i president people union souter american |
    | 56 | percent year billion poll orders economy increase october rates rose |
    | 38 | year i percent today report first program market get two |
    | 24 | i south people nea state think court frohnmayer year victims |

