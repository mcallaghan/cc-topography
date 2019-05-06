
import matplotlib
matplotlib.use('agg')

import django, sys, os, random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

matplotlib.rcParams["figure.figsize"] = [7.2,4.5]
matplotlib.rcParams['axes.labelsize'] = 7
matplotlib.rcParams['xtick.labelsize'] = 5
matplotlib.rcParams['ytick.labelsize'] = 5

sys.path.append('/home/galm/software/django/tmv/BasicBrowser/')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BasicBrowser.settings")
django.setup()
from django.db.models import Q, F, Sum, Count, FloatField, Case, When, Value, Max
from scipy.sparse import coo_matrix, find
from sklearn.manifold import TSNE

from scoping.models import *
from tmv_app.models import *
from scipy.sparse import csr_matrix
from time import time
from MulticoreTSNE import MulticoreTSNE as mTSNE





def get_matrix(run_id,s_size):
    stat = RunStats.objects.get(pk=run_id)
    if stat.method=="DT":
        dts = DocDynamicTopic.objects
    else:
        dts = DocTopic.objects

    db_matrix = dts.filter(run_id=run_id,score__gt=0.01)
    docs = set(db_matrix.values_list('doc__id',flat=True))

    if s_size ==0:
        s_docs = docs
    else:
        s_docs = random.sample(docs,s_size)

    db_matrix = dts.filter(
        run_id=run_id,
        score__gt=0.01,
        doc__id__in=s_docs
    )

    vs = list(db_matrix.values('score','doc_id','topic_id'))

    c_ind = np.array(list(set(db_matrix.values_list('topic_id',flat=True).order_by('doc_id'))))
    r_ind = np.array(list(set(db_matrix.values_list('doc_id',flat=True).order_by('doc_id'))))

    d = [x['score'] for x in vs]
    c = [int(np.where(c_ind==x['topic_id'])[0]) for x in vs]
    r = [int(np.where(r_ind==x['doc_id'])[0]) for x in vs]

    m = csr_matrix((d,(r,c)),shape=(len(r_ind),len(c_ind)))

    return(m,c_ind,r_ind)

def get_tsne(m,p):
    tsne = TSNE(n_components=2, verbose=0, perplexity=p)
    return tsne.fit_transform(m.toarray())

def draw_simple(results,r_ind,fname=None):
    cs = []
    sizes = []
    xs = []
    ys = []


    fig = plt.figure(dpi=188)

    for i,did in enumerate(r_ind):
        x = tsne_results[i,0]
        y = tsne_results[i,1]
        xs.append(x)
        ys.append(y)

    plt.scatter(
        xs,
        ys,
        s=4,
        linewidth=0.1,
        #s=sizes,
        c="#F0F0F0",
        edgecolor='k'
    )
    if fname is not None:
        plt.savefig(fname)
        plt.close()

for run_id in [1810,1811,1809,1818,1817,1814]:#,758]:
    #for s_size in [10000,20000,50000,100000,0]:
    for s_size in [0]:
        print(s_size)
        m, c_ind, r_ind = get_matrix(run_id,s_size)
        print("got m")
        np.save(
            '../tsne_results/data/run_{}_s_{}_m.npy'.format(
                run_id,
                s_size
            ),
            m
        )
        np.save(
            '../tsne_results/data/run_{}_s_{}_r_ind.npy'.format(
                run_id,
                s_size
            ),
            r_ind
        )
        #for p in [40,50,60,70,90,150]:
        for p in [20,50,100,200]:
            fname = "../tsne_results/plots/run_{}_s_{}_p_{}.png".format(
                run_id,
                s_size,
                p
            )
            print("Doing tsne with run {}, on {} docs, with {} perplexity".format(run_id,s_size,p))
            f = Path(fname)
            if f.exists():
                print("EXISTS")
                continue

            print("multicore")
            t0 = time()
            tsne = mTSNE(n_components=2, verbose=0, perplexity=p,n_jobs=4)
            tsne_results = tsne.fit_transform(m.toarray())
            print("done in %0.3fs." % (time() - t0))

            np.save(
                '../tsne_results/data/run_{}_s_{}_p{}.npy'.format(
                    run_id,
                    s_size,
                    p
                ),
                tsne_results
            )
            draw_simple(tsne_results,r_ind,fname)
