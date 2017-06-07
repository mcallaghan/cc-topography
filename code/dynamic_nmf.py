import pickle, string, numpy, getopt, sys, random, time, re, pprint, gc, resource
import pandas as pd
import onlineldavb, scrapeWoS, gensim, nltk, subprocess, psycopg2
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
from multiprocess import Pool
import django
from functools import partial
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from time import time
from django.utils import timezone
from scipy.sparse import csr_matrix, find
import numpy as np


sys.stdout.flush()

# import file for easy access to browser database
sys.path.append('/home/galm/software/tmv/BasicBrowser/')

# sys.path.append('/home/max/Desktop/django/BasicBrowser/')
import db as db
from tmv_app.models import *
from scoping.models import Doc, Query
from django.db import connection, transaction
cursor = connection.cursor()

def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

def f_gamma(docs,gamma,docsizes,docUTset,topic_ids):
    dts = []
    for d in docs:
        if gamma[2][d] > 0.005:
            dt = DocTopic(
                doc_id = docUTset[gamma[0][d]],
                topic_id = topic_ids[gamma[1][d]],
                score = gamma[2][d],
                scaled_score = gamma[2][d] / docsizes[gamma[0][d]],
                run_id_id=run_id
            )
        dts.append(dt)
    return dts

def f_gamma2(docs,gamma,docsizes,docUTset,topic_ids):
    vl = []
    for d in docs:
        if gamma[2][d] > 0.001:
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

def update_features(id):
    django.db.connections.close_all()
    term = Term.objects.get(pk=id)
    term.run_id.add(run_id)
    django.db.connections.close_all()
    return term.pk

def bulk_create_par(dts):
    DocTopic.objects.bulk_create(dts)

class snowball_stemmer(object):
    def __init__(self):
        self.stemmer = SnowballStemmer("english")
    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in tokenize(doc)]

def proc_docs(docs):
    stoplist = set(nltk.corpus.stopwords.words("english"))
    stoplist.add('elsevier')
    stoplist.add('rights')
    stoplist.add('reserved')
    stoplist.add('john')
    stoplist.add('wiley')
    stoplist.add('sons')
    stoplist.add('copyright')

    abstracts = [re.split("\([C-c]\) [1-2][0-9]{3} Elsevier",x.content)[0] for x in docs.iterator()]
    abstracts = [x.split("Published by Elsevier")[0] for x in abstracts]
    abstracts = [x.split("Copyright (C)")[0] for x in abstracts]
    abstracts = [re.split("\. \(C\) [1-2][0-9]{3} ",x)[0] for x in abstracts]
    docsizes = [len(x) for x in abstracts]
    ids = [x.UT for x in docs.iterator()]

    return [abstracts, docsizes, ids, stoplist]

def main():
    qid = 1263
    K = 10
    n_features = 50000
    n_samples = 1000
    ng = 1
    yrange=list(range(2010,2015))



    global run_id
    run_id = db.init(n_features,ng)
    stat = RunStats.objects.get(run_id=run_id)
    stat.query = Query.objects.get(pk=qid)
    stat.method = "DT"
    stat.save()
    i = 0
    for y in yrange:

        docs = Doc.objects.filter(query=qid,content__iregex='\w',PY=y)
        abstracts, docsizes, ids, stoplist = proc_docs(docs)

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

        # Get the vocab, add it to db
        vocab = tfidf_vectorizer.get_feature_names()
        vocab_ids = []
        pool = Pool(processes=8)
        vocab_ids.append(pool.map(add_features,vocab))
        pool.terminate()
        del vocab
        vocab_ids = vocab_ids[0]

        django.db.connections.close_all()
        topic_ids = db.add_topics(K)
        for t in topic_ids:
            top = Topic.objects.get(pk=t)
            top.year = y
            top.save()

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

        gamma =  find(csr_matrix(nmf.transform(tfidf)))
        glength = len(gamma[0])

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

            add_t0 = time()
            values_list = [item for sublist in values_list for item in sublist]
            pool = Pool(processes=ps)
            pool.map(insert_many,values_list)
            pool.terminate()
            add_t += time() - add_t0
            gc.collect()
            sys.stdout.flush()

        i+=1

    # django.db.connections.close_all()
    # stats = RunStats.objects.get(run_id=run_id)
    # stats.error = nmf.reconstruction_err_
    # stats.errortype = "Frobenius"
    # stats.iterations = nmf.n_iter_
    # stats.method = "nm"
    # stats.last_update=timezone.now()
    # stats.query=Query.objects.get(pk=qid)
    # stats.save()
    # django.db.connections.close_all()

    subprocess.Popen(["python3",
        "/home/galm/software/tmv/BasicBrowser/update_all_topics.py",
        str(run_id)
    ]).wait()

if __name__ == '__main__':
    t0 = time()
    main()
    totalTime = time() - t0

    tm = int(totalTime//60)
    ts = int(totalTime-(tm*60))

    print("done! total time: " + str(tm) + " minutes and " + str(ts) + " seconds")
    print("a maximum of " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000) + " MB was used")
