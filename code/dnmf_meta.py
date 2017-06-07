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

def f_dlambda(t,m,v_ids,t_ids):
    tt = DynamicTopicTerm(
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
    run_id = 124

    DynamicTopic.objects.filter(run_id=run_id).delete()

    tops = Topic.objects.filter(run_id=run_id)
    terms = Term.objects.all()

    B = numpy.zeros((tops.count(),terms.count()))

    print(tops)

    wt = 0
    for topic in tops:
        tts = TopicTerm.objects.filter(
            topic=topic, run_id=run_id,
            score__gt=0.00001
        ).order_by('-score')[:100]
        for tt in tts:
            B[wt,tt.term.id] = tt.score
        wt+=1

    col_sum = np.sum(B,axis=0)
    vocab_ids = np.flatnonzero(col_sum)

    print(B)

    B = B[:,vocab_ids]

    print(B)

    nmf = NMF(
        n_components=K, random_state=1,
        alpha=.1, l1_ratio=.5
    ).fit(B)


    ## Add dynamic topics
    dtopics = []
    for k in range(K):
        dtopic = DynamicTopic(
            run_id=RunStats.objects.get(pk=run_id)
        )
        dtopic.save()
        dtopics.append(dtopic)

    dtopic_ids = list(DynamicTopic.objects.filter(run_id=run_id).values_list('id',flat=True))

    print(dtopic_ids)

    ##################
    ## Add the dtopic*term matrix to the db
    print("Adding topicterms to db")
    t0 = time()
    ldalambda = find(csr_matrix(nmf.components_))
    topics = range(len(ldalambda[0]))
    tts = []
    pool = Pool(processes=8)
    tts.append(pool.map(partial(f_dlambda, m=ldalambda,
                    v_ids=vocab_ids,t_ids=dtopic_ids),topics))
    pool.terminate()
    tts = flatten(tts)
    gc.collect()
    sys.stdout.flush()
    django.db.connections.close_all()
    DynamicTopicTerm.objects.bulk_create(tts)
    print("done in %0.3fs." % (time() - t0))

    ## Add the wtopic*dtopic matrix to the database
    gamma = nmf.transform(B)

    for topic in range(len(gamma)):
        for dtopic in range(len(gamma[topic])):
            if dtopic > 0:
                tdt = TopicDTopic(
                    topic = tops[topic],
                    dynamictopic = dtopics[dtopic],
                    score = gamma[topic][dtopic]
                )
                tdt.save()

    # Calculate the primary dtopic for each topic
    for t in tops:
        t.primary_dtopic = TopicDTopic.objects.filter(
            topic=t
        ).order_by('-score').first().dynamictopic
        t.save()




if __name__ == '__main__':
    t0 = time()
    main()
    totalTime = time() - t0

    tm = int(totalTime//60)
    ts = int(totalTime-(tm*60))

    print("done! total time: " + str(tm) + " minutes and " + str(ts) + " seconds")
    print("a maximum of " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000) + " MB was used")
