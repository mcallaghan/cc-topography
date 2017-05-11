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

import pickle, string, numpy, getopt, sys, random, time, re, pprint, gc, resource
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
from scipy.sparse import csr_matrix, find
import numpy as np
import psycopg2

#conn = psycopg2.connect("dbname=tmv_app user=tmv password=topicmodels")

sys.stdout.flush()

# import file for easy access to browser database
sys.path.append('/home/galm/software/tmv/BasicBrowser/')

# sys.path.append('/home/max/Desktop/django/BasicBrowser/')
import db as db
from tmv_app.models import *
from scoping.models import Doc
from django.db import connection, transaction
cursor = connection.cursor()

def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

# def f_gamma(d,gamma,docsizes,docUTset,topic_ids):
#     dts = []
#     django.db.connections.close_all()
#     doc_size = docsizes[d]
#     doc_id = docUTset[d]
#     row = gamma[d]
#     nzs = find(row)
#     for nz in range(len(nzs[1])):
#         k = nzs[1][nz]
#         score = nzs[2][nz]
#         doc = Doc.objects.get(UT=doc_id)
#         topic = Topic.objects.get(pk=topic_ids[k])
#         dt = DocTopic(
#             doc=doc, topic=topic, score=score,
#             scaled_score=score,
#             run_id=run_id
#         )
#         dts.append(dt)
#         # db.add_doc_topic_sk(doc_id, topic_ids[k], gamma[d][k], gamma[d][k]/doc_size)
#     return dts
#     django.db.connections.close_all()

def f_gamma(docs,gamma,docsizes,docUTset,topic_ids):
    dts = []
    for d in docs:
        dt = DocTopic(
            doc_id = docUTset[gamma[0][d]],
            topic_id = topic_ids[gamma[1][d]],
            score = gamma[2][d],
            scaled_score = gamma[2][d] / docsizes[gamma[0][d]],
            run_id=run_id
        )
        dts.append(dt)
    return dts

def f_gamma2(docs,gamma,docsizes,docUTset,topic_ids):
    vl = []
    for d in docs:
        dt = (
            docUTset[gamma[0][d]],
            topic_ids[gamma[1][d]],
            gamma[2][d],
            gamma[2][d] / docsizes[gamma[0][d]],
            run_id
        )
        vl.append(dt)
    return vl

def f_lambda(t,m,v_ids,t_ids):
    tt = TopicTerm(
        term_id = v_ids[m[1][t]],
        topic_id = t_ids[m[0][t]],
        score = m[2][t],
        run_id = run_id
    )
    return tt

def tokenize(text):
    transtable = {ord(c): None for c in string.punctuation + string.digits}
    tokens = nltk.word_tokenize(text.translate(transtable))
    tokens = [i for i in tokens if len(i) > 2]
    return tokens


def add_features(title):
    django.db.connections.close_all()
    term, created = Term.objects.get_or_create(title=title)
    term.run_id.add(run_id)
    django.db.connections.close_all()
    return term.pk

def bulk_create_par(dts):
    DocTopic.objects.bulk_create(dts)

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
        print(ng)
    except:
        ng = 1
    try:
        n_features = int(sys.argv[3])
        print(n_features)
    except:
        n_features = 50000
    try:
        limit = int(sys.argv[4])
    except:
        limit = False

    print("###################################\nStarting \
    NMF.py with K={}, ngrame={} and n_features={}".format(K,ng,n_features))

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

    #docs = Doc.objects.filter(query=893,content__iregex='\w').values('UT','title','content')
    docs = Doc.objects.filter(query=365,content__iregex='\w') | Doc.objects.filter(query=354,content__iregex='\w')
    docs = docs#.values('UT','content')
    if limit is not False:
        docs = docs[:limit]

    print("{} documents found".format(len(docs)))
    # abstracts = [re.split("\([C-c]\) [1-2][0-9]{3} Elsevier",x['content'])[0] for x in docs]
    # abstracts = [x.split("Published by Elsevier")[0] for x in abstracts]
    # abstracts = [x.split("Copyright (C)")[0] for x in abstracts]
    # abstracts = [re.split("\. \(C\) [1-2][0-9]{3} ",x)[0] for x in abstracts]
    # ids = [x['UT'] for x in docs]

    # use DOCS ITERATOR?????

    abstracts = [re.split("\([C-c]\) [1-2][0-9]{3} Elsevier",x.content)[0] for x in docs.iterator()]
    #abstracts = [x.split("Published by Elsevier")[0] for x in abstracts]
    #abstracts = [x.split("Copyright (C)")[0] for x in abstracts]
    #abstracts = [re.split("\. \(C\) [1-2][0-9]{3} ",x)[0] for x in abstracts]
    docsizes = [len(x) for x in abstracts]
    ids = [x.UT for x in docs.iterator()]

    del docs
    class snowball_stemmer(object):
        def __init__(self):
            self.stemmer = SnowballStemmer("english")
        def __call__(self, doc):
            return [self.stemmer.stem(t) for t in tokenize(doc)]


    global run_id
    run_id = db.init(n_features,ng)

    #############################################
    # Use tf-idf features for NMF.
    print("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=10,
                                       max_features=n_features,
                                       ngram_range=(ng,ng),
                                       tokenizer=snowball_stemmer(),
                                       stop_words=stoplist)
    t0 = time()
    tfidf = tfidf_vectorizer.fit_transform(abstracts)
    print("done in %0.3fs." % (time() - t0))

    del abstracts

    gc.collect()

    vocab = tfidf_vectorizer.get_feature_names()

    print(len(vocab))

    # add terms to db
    print("Adding features to db")
    t0 = time()
    vocab_ids = []
    pool = Pool(processes=8)
    vocab_ids.append(pool.map(add_features,vocab))
    pool.terminate()
    print("done in %0.3fs." % (time() - t0))

    vocab_ids = vocab_ids[0]

    # add empty topics to db
    django.db.connections.close_all()
    topic_ids = db.add_topics(K)

    gc.collect()

    # Fit the NMF model
    print("Fitting the NMF model with tf-idf features, "
          "n_samples=%d and n_features=%d..."
          % (n_samples, n_features))
    t0 = time()
    nmf = NMF(n_components=K, random_state=1,
              alpha=.1, l1_ratio=.5).fit(tfidf)
    print("done in %0.3fs." % (time() - t0))


    print("Adding topicterms to db")
    t0 = time()
    ldalambda = find(csr_matrix(nmf.components_))
    topics = range(len(ldalambda[0]))
    tts = []
    pool = Pool(processes=8)

    tts.append(pool.map(partial(f_lambda, m=ldalambda,
                    v_ids=vocab_ids,t_ids=topic_ids),topics))
    pool.terminate()
    tts = flatten(tts)
    gc.collect()
    sys.stdout.flush()
    django.db.connections.close_all()
    TopicTerm.objects.bulk_create(tts)
    print("done in %0.3fs." % (time() - t0))


    t0 = time()
    print("making sparse matrix")
    gamma =  find(csr_matrix(nmf.transform(tfidf)))

    # Make the gamma longer to test memory performance with big sets
    gamma = [np.concatenate([x,x,x,x,x,x,x,x,x,x]) for x in gamma]
    gamma = [np.concatenate([x,x,x,x,x,x,x,x,x,x]) for x in gamma]
    gamma = [np.concatenate([x,x,x,x,x,x,x,x,x,x]) for x in gamma]
    #gamma = [np.concatenate([x,x,x,x,x,x,x,x,x,x]) for x in gamma]
    print("done in %0.3fs." % (time() - t0))
    glength = len(gamma[0])

    t0 = time()
    print("adding {} doctopics to the db".format(glength))
    chunk_size = 100000
    ps = 16
    parallel_add = True

    all_dts = []

    make_t = 0
    add_t = 0

    def insert_many(values_list):
        query='''
            INSERT INTO "tmv_app_doctopic"
            ("doc_id", "topic_id", "score", "scaled_score", "run_id")
            VALUES (%s,%s,%s,%s,%s)
        '''
        cursor = connection.cursor()
        cursor.executemany(query,values_list)



    for i in range(glength//chunk_size+1):
        dts = []
        values_list = []
        f = i*chunk_size
        l = (i+1)*chunk_size
        if l > glength:
            l = glength
        docs = range(f,l)
        doc_batches = []
        for p in range(ps):
            doc_batches.append([x for x in docs if x % ps == p])
        pool = Pool(processes=ps)
        make_t0 = time()
        values_list.append(pool.map(partial(f_gamma2, gamma=gamma,
                        docsizes=docsizes,docUTset=ids,topic_ids=topic_ids),doc_batches))
        #dts.append(pool.map(partial(f_gamma, gamma=gamma,
        #                docsizes=docsizes,docUTset=ids,topic_ids=topic_ids),doc_batches))
        pool.terminate()
        make_t += time() - make_t0
        django.db.connections.close_all()
        if not parallel_add:
            dts = flatten(dts)
            values_list = [item for sublist in values_list for item in sublist]
            values_list = [item for sublist in values_list for item in sublist]
            add_t0 = time()
            #DocTopic.objects.bulk_create(dts)
            query='''
                INSERT INTO "tmv_app_doctopic"
                ("doc_id", "topic_id", "score", "scaled_score", "run_id")
                VALUES (%s,%s,%s,%s,%s)
            '''
            cursor = connection.cursor()
            cursor.executemany(query,values_list)

            add_t += time() - add_t0
        else:
            add_t0 = time()
            #dts = [item for sublist in dts for item in sublist]
            values_list = [item for sublist in values_list for item in sublist]
            pool = Pool(processes=ps)
            #pool.map(bulk_create_par, dts)
            pool.map(insert_many,values_list)
            pool.terminate()
            add_t += time() - add_t0
        gc.collect()
        sys.stdout.flush()

    print("done in %0.3fs." % (time() - t0))
    print("Making the objects took in %0.3fs." % (make_t))
    print("Adding the objects took in %0.3fs." % (add_t) )

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
    print("a maximum of " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000) + " MB was used")
