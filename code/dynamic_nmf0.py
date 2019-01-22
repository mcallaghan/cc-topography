import pickle, string, numpy, getopt, sys, random, time, re, pprint, gc, resource
import pandas as pd
import nltk, subprocess, psycopg2, math
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
from django.core import management
import os



sys.stdout.flush()

# import file for easy access to browser database
sys.path.append('/home/galm/software/django/tmv/BasicBrowser')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BasicBrowser.settings")
django.setup()

# sys.path.append('/home/max/Desktop/django/BasicBrowser/')
import utils.db as db
from tmv_app.models import *
from scoping.models import *
from django.db import connection, transaction
from django.db.models import Count, Sum
from utils.text import *
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
        if gamma[2][d] > 0:
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


def predict(x):
    a = 43.78
    b1 = 0.006912
    b2 = -0.0000001567
    x_hat = a+b1*x+b2*x*x
    if x_hat > x:
        x_hat=x/2
    return(int(round(x_hat)))

def predict(x):
    a = -109
    b = 22.438
    #b = 10
    x_hat = a+b*np.log(x)
    if x_hat > x:
        x_hat=x/2
    if x_hat < 2:
        x_hat=2
    return(int(round(x_hat)))

def predict(x):
    a = -75
    b = 14.5
    #b = 10
    x_hat = a+b*np.log(x)
    if x_hat > x:
        x_hat=x/2
    if x_hat < 2:
        x_hat=2
    return(int(round(x_hat)))

def main():
    try:
        qid = int(sys.argv[1])
    except:
        qid = 3769
    # The n in ngram
    try:
        K = int(sys.argv[2])
        print(K)
    except:
        K = 100

    n_features = 50000
    n_samples = 1000
    ng = 1
    yrange=list(range(1990,2018))

    Ks = [10, 25, 40, 60, 80, 100]


    global run_id
    run_id = db.init(n_features)
    stat = RunStats.objects.get(run_id=run_id)
    stat.query = Query.objects.get(pk=qid)
    stat.method = "DT"
    stat.alpha = 0.01
    stat.save()
    i = 0
    #ndocs = Doc.objects.filter(query=qid,content__iregex='\w').count()
    #avdocs = ndocs/len(yrange)
    docs = Doc.objects.filter(query=qid,relevant=True,content__iregex='\w')
    avdocs = docs.filter(PY=2016).count()
    print(avdocs)

    from nltk import wordpunct_tokenize
    from nltk import WordNetLemmatizer
    from nltk import sent_tokenize
    from nltk import pos_tag
    import string
    from nltk.corpus import stopwords as sw
    punct = set(string.punctuation)
    from nltk.corpus import wordnet as wn
    stopwords = set(sw.words('english'))

    def lemmatize(token, tag):
            tag = {
                'N': wn.NOUN,
                'V': wn.VERB,
                'R': wn.ADV,
                'J': wn.ADJ
            }.get(tag[0], wn.NOUN)
            return WordNetLemmatizer().lemmatize(token, tag)

    dlen = docs.count()

    kws = docs.filter(
        kw__text__iregex='\w+[\-\ ]'
    ).values('kw__text').annotate(
        n = Count('pk')
    ).filter(n__gt=dlen//1000).order_by('-n')

    kw_text = set([x['kw__text'].replace('-',' ') for x in kws])
    kw_ws = set([x['kw__text'].replace('-',' ').split()[0] for x in kws]) - stopwords

    for iteration, ar in enumerate(AR.objects.filter(ar__gt=0).order_by('ar')):
        ys = list(range(ar.start,ar.end+1))
        y = ar.ar

        print(ys)

        tp, created = TimePeriod.objects.get_or_create(
            title=ar.name,
            n = ar.ar,
            ys = list(ys)
        )
        tp.save()

        stat.periods.add(tp)

    #for y in yrange:
        #ys = [y]

        docs = Doc.objects.filter(query=qid,content__iregex='\w',PY__in=ys)

        ydocs = docs.count()
        print("\n#######################")
        print("IN YEAR {}: {} docs".format(y,ydocs))
        k = Ks[iteration]
        print("esimating {} topics...".format(k))

        abstracts, docsizes, ids, citations = proc_docs(docs, stoplist)
        #abstracts, docsizes, ids, stoplist = proc_docs(docs,stoplist)

        def fancy_tokenize(X):

            common_words = set([x.lower() for x in X.split()]) & kw_ws
            for w in list(common_words):
                w = w.replace('(','').replace(')','')
                wpat = "({}\W*\w*)".format(w)
                wn = [x.lower().replace('-',' ') for x in re.findall(wpat, X, re.IGNORECASE)]
                kw_matches = set(wn) & kw_text
                if len(kw_matches) > 0:
                    for m in kw_matches:
                        insensitive_m = re.compile(m, re.IGNORECASE)
                        X = insensitive_m.sub(' ', X)
                        yield m.replace(" ","-")

            for sent in sent_tokenize(X):
                for token, tag in pos_tag(wordpunct_tokenize(sent)):
                    token = token.lower().strip()
                    if token in stopwords:
                        continue
                    if all(char in punct for char in token):
                        continue
                    if len(token) < 3:
                        continue
                    if all(char in string.digits for char in token):
                        continue
                    lemma = lemmatize(token,tag)
                    yield lemma

        tokenizer = fancy_tokenize

        #############################################
        # Use tf-idf features for NMF.
        print("Extracting tf-idf features for NMF...")
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=5,
                                           max_features=n_features,
                                           ngram_range=(ng,ng),
                                           tokenizer=fancy_tokenize,
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
        topic_ids = db.add_topics(k, run_id)
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
        nmf = NMF(n_components=k, random_state=1,
                  alpha=stat.alpha, l1_ratio=.5).fit(tfidf)
        print("done in %0.3fs." % (time() - t0))


        print("Adding topicterms to db")
        t0 = time()
        ldalambda = find(csr_matrix(nmf.components_))
        topics = range(len(ldalambda[0]))
        tts = []
        pool = Pool(processes=6)

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

        ps = 6
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

        stat.error = stat.error + nmf.reconstruction_err_
        stat.errortype = "Frobenius"

    ## After all the years have been run, update the dtops

    K = K


    tops = Topic.objects.filter(run_id=run_id)
    terms = Term.objects.all()

    B = numpy.zeros((tops.count(),terms.count()))

    print(tops)

    #

    wt = 0
    for topic in tops:
        tts = TopicTerm.objects.filter(
            topic=topic
        ).order_by('-score')[:50]
        for tt in tts:
            B[wt,tt.term.id] = tt.score
        wt+=1

    col_sum = np.sum(B,axis=0)
    vocab_ids = np.flatnonzero(col_sum)

    # we only want the columns where there are at least some
    # topic-term values
    B = B[:,vocab_ids]


    nmf = NMF(
        n_components=K, random_state=1,
        alpha=stat.alpha, l1_ratio=.5
    ).fit(B)


    ## Add dynamic topics
    dtopics = []
    for k in range(K):
        dtopic = DynamicTopic(
            run_id=RunStats.objects.get(pk=run_id)
        )
        dtopic.save()
        dtopics.append(dtopic)

    dtopic_ids = list(
        DynamicTopic.objects.filter(
            run_id=run_id
        ).values_list('id',flat=True)
    )

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
            if gamma[topic][dtopic] > 0:
                tdt = TopicDTopic(
                    topic = tops[topic],
                    dynamictopic_id = dtopic_ids[dtopic],
                    score = gamma[topic][dtopic]
                )
                tdt.save()

    ## Calculate the primary dtopic for each topic

    for t in tops:
        try:
            t.primary_dtopic.add(TopicDTopic.objects.filter(
                topic=t
            ).order_by('-score').first().dynamictopic)
            t.save()
        except:
            pass


    stat.error = stat.error + nmf.reconstruction_err_
    stat.errortype = "Frobenius"
    stat.last_update=timezone.now()
    stat.save()
    management.call_command('update_run',run_id)






if __name__ == '__main__':
    t0 = time()
    main()
    totalTime = time() - t0

    tm = int(totalTime//60)
    ts = int(totalTime-(tm*60))

    print("done! total time: " + str(tm) + " minutes and " + str(ts) + " seconds")
    print("a maximum of " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000) + " MB was used")
