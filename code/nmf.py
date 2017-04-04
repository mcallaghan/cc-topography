#!/usr/bin/env python3

# onlinewikipedia.py: Demonstrates the use of online VB for LDA to
# analyze a bunch of random Wikipedia articles.
#
# Copyright (C) 2010  Matthew D. Hoffman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import pickle, string, numpy, getopt, sys, random, time, re, pprint, gc
import pandas as pd
import onlineldavb
import scrapeWoS
import gensim
import nltk
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
import sys
import time
from multiprocess import Pool
import django
from functools import partial
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from time import time
from django.utils import timezone

sys.stdout.flush()

# import file for easy access to browser database
sys.path.append('/home/galm/software/tmv/BasicBrowser/')

# sys.path.append('/home/max/Desktop/django/BasicBrowser/')
import db3 as db
from tmv_app.models import *


def main():

    # The number of topics
    try:
        K = int(sys.argv[1])
        print(K)
    except:
        K = 40
    # The n in ngram
    try:
        ng = int(sys.argv[2])
        print(K)
    except:
        ng = 1

    n_features = 50000
    n_samples = 1000

    #############################################
    ## STOPWORDS
    stoplist = set(nltk.corpus.stopwords.words("english"))
    stoplist.add('elsevier')
    stoplist.add('rights')
    stoplist.add('reserved')
    stoplist.add('john')
    stoplist.add('wiley')
    stoplist.add('sons')
    stoplist.add('copyright')

    docs = Doc.objects.all().exclude(UT__contains='WOS2:').exclude(UT__contains='2WOS').filter(content__iregex='\w').values('UT','title','content')

    print(len(docs))
    abstracts = [re.split("\(C\) [1-2][0-9]{3} Elsevier",x['content'])[0] for x in docs]
    abstracts = [x.split("Published by Elsevier")[0] for x in abstracts]
    abstracts = [x.split("Copyright (C)")[0] for x in abstracts]
    abstracts = [re.split("\. \(C\) [1-2][0-9]{3} ",x)[0] for x in abstracts]
    titles = [x['title'] for x in docs]
    ids = [x['UT'] for x in docs]

    def tokenize(text):
        transtable = {ord(c): None for c in string.punctuation + string.digits}
        tokens = nltk.word_tokenize(text.translate(transtable))
        tokens = [i for i in tokens if len(i) > 2]
        return tokens        

    class snowball_stemmer(object):
        def __init__(self):
            self.stemmer = SnowballStemmer("english")
        def __call__(self, doc):
            return [self.stemmer.stem(t) for t in tokenize(doc)]
    


    run_id = db.init()
    
    #############################################
    # Use tf-idf features for NMF.
    print("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=1,
                                       max_features=n_features,
                                       ngram_range=(ng,ng),
                                       tokenizer=snowball_stemmer(),
                                       stop_words=stoplist)
    t0 = time()
    tfidf = tfidf_vectorizer.fit_transform(abstracts)
    print("done in %0.3fs." % (time() - t0))


    vocab = tfidf_vectorizer.get_feature_names()

    sys.exit()

    print(len(vocab))

    

    # add terms to db
    vocab_ids = db.add_features(vocab)

    # add empty topics to db
    topic_ids = db.add_topics(K)

    # add all docs
    def f_doc(d):
        django.db.connections.close_all()
        db.add_doc(d[0],d[1],d[2],d[3],d[4],d[5])
        django.db.connections.close_all()

    def f_gamma(d,gamma,docset,docUTset,topic_ids):
        django.db.connections.close_all()
        doc_size = len(docset[d])
        doc_id = docUTset[d]
        for k in range(len(gamma[d])):
            db.add_doc_topic_sk(doc_id, topic_ids[k], gamma[d][k], gamma[d][k]/doc_size)
        django.db.connections.close_all()

    def f_lambda(topic_no,ldalambda,vocab_ids,topic_ids):
        django.db.connections.close_all()
        lambda_sum = sum(ldalambda[topic_no])
        db.clear_topic_terms(topic_no)
        for term_no in range(len(ldalambda[topic_no])):
            term_id = vocab_ids[term_no]
            topic_id = topic_ids[topic_no]
            db.add_topic_term_sk(topic_id, term_id, ldalambda[topic_no][term_no]/lambda_sum)
        django.db.connections.close_all()

    gc.collect()

    # Fit the NMF model
    print("Fitting the NMF model with tf-idf features, "
          "n_samples=%d and n_features=%d..."
          % (n_samples, n_features))
    t0 = time()
    nmf = NMF(n_components=K, random_state=1,
              alpha=.1, l1_ratio=.5).fit(tfidf)
    print("done in %0.3fs." % (time() - t0))


    topics = range(len(nmf.components_))
    pool = Pool(processes=8)
    pool.map(partial(f_lambda, ldalambda=nmf.components_,
                    vocab_ids=vocab_ids,topic_ids=topic_ids),topics) 
    pool.terminate()

    gamma = nmf.transform(tfidf)
    docs = range(len(gamma))

    pool = Pool(processes=8)
    pool.map(partial(f_gamma, gamma=gamma, 
                    docset=abstracts, docUTset=ids,topic_ids=topic_ids),docs)
    pool.terminate()

    gc.collect()
    sys.stdout.flush()

    django.db.connections.close_all()
    stats = RunStats.objects.get(run_id=run_id)
    stats.error = nmf.reconstruction_err_
    stats.errortype = "Frobenius"
    stats.iterations = nmf.n_iter_
    stats.method = "nm"
    stats.last_update=timezone.now()
    stats.save()
    django.db.connections.close_all()

if __name__ == '__main__':
    t0 = time()
    main()
    totalTime = time() - t0

    tm = int(totalTime//60)
    ts = int(totalTime-(tm*60))

    print("done! total time: " + str(tm) + " minutes and " + str(ts) + " seconds")
