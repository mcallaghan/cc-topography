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
from time import time, sleep
from django.utils import timezone
from scipy.sparse import csr_matrix, find
import numpy as np
from django.core import management



sys.stdout.flush()

# import file for easy access to browser database
sys.path.append('/home/galm/software/django/tmv/BasicBrowser')

# sys.path.append('/home/max/Desktop/django/BasicBrowser/')
import db as db
from tmv_app.models import *
from scoping.models import *

from scoping.tasks import *
from tmv_app.tasks import *

def predict(x):
    a = -70
    b = 15
    #b = 10
    x_hat = a+b*np.log(x)
    if x_hat > x:
        x_hat=x/2
    if x_hat < 2:
        x_hat=2
    return(int(round(x_hat)))


q = Query.objects.get(pk=365)

docs = Doc.objects.filter(query=q,content__iregex='\w')

for ar in AR.objects.filter(ar__gt=0):
    ys = range(ar.start,ar.end+1)
    y = ar.ar

    print(ar.name)

    p = q.project

    u = User.objects.get(username="galm")

    nq, created = Query.objects.get_or_create(
        title="climate all {}".format(ar.name),
        type="default",
        text="MANUALLY GET CC DOCS FROM AR",
        project=p,
        creator = u,
        #date = timezone.now(),
        database = "intern"
    )
    nq.save()

    if created:
        ydocs = docs.filter(PY__in=ys)
        for d in ydocs.iterator():
            d.query.add(nq)
        nq.r_count=ydocs.count()
        nq.date = timezone.now()
        nq.save()

    k = predict(nq.r_count)
    for K in [k-20,k-10,k,k+10,k+20,k+30]:
        stat, created = RunStats.objects.get_or_create(
            min_freq=5,
            K=K,
            alpha=0.1,
            method="NM",
            query=nq
        )

        stat.save()

        if stat.status == 3:
            continue

        wait=True
        while wait==True:
            if os.getloadavg()[0] > 5:
                sleep(60)
            else:
                wait = False

        print(stat.query)
        print(stat.K)
        do_nmf.delay(stat.run_id)

        sleep(180)
