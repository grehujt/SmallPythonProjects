
import os
import scipy as sp

trainDir = './data/20news-bydate-train'
testDir = './data/20news-bydate-test'


def load_20_news_groups(d):
    data_train = {}
    for parent, folder, files in os.walk(d):
        print parent, folder, len(files)

load_20_news_groups(trainDir)
