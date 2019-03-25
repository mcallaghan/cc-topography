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
from tmv_app.utils.blei import readInfo, dtm_topic
cursor = connection.cursor()


def main():
    try:
        qid = int(sys.argv[1])
    except:
        qid = 6099
    # The n in ngram
    try:
        K = int(sys.argv[2])
        print(K)
    except:
        K = 100
    try:
        sample = int(sys.argv[3])
        print(sample)
    except:
        sample = False

    try:
        alpha = int(sys.argv[4])
        print(alpha)
    except:
        alpha = 0.2

    try:
        top_chain_var = int(sys.argv[5])
        print(sample)
    except:
        top_chain_var = 0.05

    try:
        no_model = sys.argv[6]
        no_model = True
    except:
        no_model = False

    print("DTM")
    dtm_path = "/home/galm/software/dtm/dtm/main"
    call_to_blei_algorithm=True

    n_features = 50000
    n_samples = 1000
    ng = 1

    q = Query.objects.get(pk=qid)

    stat = RunStats(
        query=q,
        method="BD",
        alpha=alpha,
        K=K,
        max_features=50000,
        top_chain_var=top_chain_var
    )
    stat.save()

    run_id = stat.pk

    ##########################
    ## create input and output folder

    input_path = './dtm-input-{}'.format(stat.pk)
    output_path = './dtm-output-{}'.format(stat.pk)

    if no_model:
        input_path = f'/var/www/files/dtm-input-{stat.pk}'

    if os.path.isdir(input_path):
        if call_to_blei_algorithm:
            shutil.rmtree(input_path)
            os.mkdir(input_path)
    else:
        os.mkdir(input_path)

    if os.path.isdir(output_path):
        if call_to_blei_algorithm:
            shutil.rmtree(output_path)
            os.mkdir(output_path)
    else:
        os.mkdir(output_path)

    docs = Doc.objects.filter(query=q, content__iregex='\w')

    if sample:
        rids = random.sample(list(docs.values_list('pk',flat=True)),sample)
        docs = Doc.objects.filter(pk__in=rids)


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

    def fancy_tokenize(X):
        X = X.lower().replace("power plant","power-plant")
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

    for iteration, ar in enumerate(AR.objects.filter(ar__gt=0).order_by('ar')):
        ys = list(range(ar.start,ar.end+1))
        tp, created = TimePeriod.objects.get_or_create(
            title=ar.name,
            n = ar.ar,
            ys = list(ys)
        )
        stat.periods.add(tp)

    time_range = sorted([tp.n for tp in stat.periods.all().order_by('n')])
    time_counts = {item.n: docs.filter(PY__in=item.ys).count() for item in stat.periods.all().order_by('n')}

    print(time_range)
    print(time_counts)

    abstracts, docsizes, ids, citations = proc_docs(docs, stoplist)

    #############################################
    # Use tf-idf features for NMF.
    print("Extracting count features for DTM...")
    vectorizer = CountVectorizer(max_df=0.95, min_df=5,
                                       max_features=stat.max_features,
                                       tokenizer=fancy_tokenize,
                                       stop_words=stoplist)
    t0 = time()
    dtm = vectorizer.fit_transform(abstracts)


    with open(os.path.join(input_path ,'foo-doctexts.dat') ,'w') as f:
        for i, text in enumerate(abstracts):
            f.write("D#{}: ".format(i) + text + "\n")
        f.write('\n')

    with open(os.path.join(input_path ,'foo-docids.dat') ,'w') as f:
        for id, ds in zip(ids, docsizes):
            f.write(f"{id}:{ds}\n")
        f.write('\n')

    del abstracts

    gc.collect()

    print("Save terms to DB")
    # Get the vocab, add it to db
    vocab = vectorizer.get_feature_names()
    vocab_ids = []
    pool = Pool(processes=8)
    vocab_ids.append(pool.map(partial(db.add_features ,run_id=run_id) ,vocab))
    pool.terminate()

    vocab_ids = vocab_ids[0]
    with open(os.path.join(input_path ,'foo-vocab.dat') ,'w') as f:
        for i, w in enumerate(vocab):
            f.write(str(vocab_ids[i]) + ": " + w + "\n")
        f.write('\n')

    del vocab

    django.db.connections.close_all()

    print("write input files for Blei algorithm")

    with open(os.path.join(input_path ,'foo-mult.dat') ,'w') as mult:
        for d in range(dtm.shape[0]):
            words = find(dtm[d])
            uwords = len(words[0])
            mult.write(str(uwords) + " ")
            for w in range(uwords):
                index = words[1][w]
                count = words[2][w]
                mult.write(str(index ) +": " +str(count ) +" ")
            mult.write('\n')

    ##########################
    ##put counts per time step in the seq file

    with open(os.path.join(input_path, 'foo-seq.dat') ,'w') as seq:
        seq.write(str(len(time_range)))

        for key, value in time_counts.items():
            seq.write('\n')
            seq.write(str(value))

    ##########################
    # Run the dtm
    if no_model:
        print("Exiting before running the model. You're probably going to run it somewhere else?")
        sys.exit()

    if call_to_blei_algorithm:
        print("Calling Blei algorithm")
        subprocess.Popen([
            dtm_path,
            "--ntopics={}".format(K),
            "--mode=fit",
            "--rng_seed=0",
            "--initialize_lda=true",
            "--corpus_prefix={}".format(os.path.join(os.path.abspath(input_path), 'foo')),
            "--outname={}".format(os.path.abspath(output_path)),
            "--top_chain_var={}".format(stat.top_chain_var),
            "--alpha={}".format(stat.alpha),
            "--lda_sequence_min_iter=10",
            "--lda_sequence_max_iter=20",
            "--lda_max_em_iter=20"
        ]).wait()
        print("Blei algorithm done")

    ##########################
    ## Upload the dtm results to the db

    print("upload dtm results to db")

    info = readInfo(os.path.join(output_path, "lda-seq/info.dat"))

    topic_ids = db.add_topics(stat.K, stat.run_id)

    #################################
    # TopicTerms

    print("writing topic terms")
    topics = range(info['NUM_TOPICS'])
    pool = Pool(processes=8)
    pool.map(partial(
        dtm_topic,
        info=info,
        topic_ids=topic_ids,
        vocab_ids=vocab_ids,
        ys = time_range,
        run_id=run_id,
        output_path=output_path
    ) ,topics)
    pool.terminate()
    gc.collect()

    ######################################
    # Doctopics
    print("writing doctopics")
    gamma = np.fromfile(os.path.join(output_path, 'lda-seq/gam.dat'), dtype=float ,sep=" ")
    gamma = gamma.reshape((int(len(gamma ) /stat.K) ,stat.K))

    gamma = find(csr_matrix(gamma))
    glength = len(gamma[0])
    chunk_size = 100000
    ps = 4
    parallel_add = True

    all_dts = []

    make_t = 0
    add_t = 0

    for i in range(glength//chunk_size +1):
        dts = []
        values_list = []
        f = i* chunk_size
        l = (i + 1) * chunk_size
        if l > glength:
            l = glength
        docs = range(f, l)
        doc_batches = []
        for p in range(ps):
            doc_batches.append([x for x in docs if x % ps == p])
        pool = Pool(processes=ps)
        make_t0 = time()
        values_list.append(pool.map(partial(db.f_gamma_batch, gamma=gamma,
                                            docsizes=docsizes, docUTset=ids,
                                            topic_ids=topic_ids, run_id=run_id),
                                    doc_batches))
        pool.terminate()
        make_t += time() - make_t0
        django.db.connections.close_all()

        add_t0 = time()
        values_list = [item for sublist in values_list for item in sublist]

        pool = Pool(processes=ps)
        pool.map(db.insert_many, values_list)
        pool.terminate()

        add_t += time() - add_t0
        gc.collect()
        sys.stdout.flush()

    stat = RunStats.objects.get(run_id=run_id)
    stat.last_update = timezone.now()
    stat.status = 3  # 3 = finished
    stat.save()
    management.call_command('update_run', run_id)

    totalTime = time() - t0

    tm = int(totalTime // 60)
    ts = int(totalTime - (tm * 60))

    print("done! total time: " + str(tm) + " minutes and " + str(ts) + " seconds")




if __name__ == '__main__':
    t0 = time()
    main()
    totalTime = time() - t0

    tm = int(totalTime//60)
    ts = int(totalTime-(tm*60))

    print("done! total time: " + str(tm) + " minutes and " + str(ts) + " seconds")
    print("a maximum of " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000) + " MB was used")
