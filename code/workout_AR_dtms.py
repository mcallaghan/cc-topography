import os, sys, time, resource, re, gc, shutil
from multiprocess import Pool
from functools import partial
from urllib.parse import urlparse, parse_qsl
import matplotlib
matplotlib.use('pgf')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import django
import pickle
from django.db.models import Count, Sum
sys.path.append('/home/galm/software/django/tmv/BasicBrowser/')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BasicBrowser.settings")
django.setup()

import utils.text as tutils

from scoping.models import *

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
import nltk
from nltk.stem import SnowballStemmer
import string

stoplist = set(nltk.corpus.stopwords.words("english"))
stoplist.add('elsevier')
stoplist.add('rights')
stoplist.add('reserved')
stoplist.add('john')
stoplist.add('wiley')
stoplist.add('sons')
stoplist.add('copyright')

from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords as sw
punct = set(string.punctuation)
from nltk.corpus import wordnet as wn
stopwords = stoplist

q = Query.objects.get(pk=6187)

def lemmatize(token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)
        return WordNetLemmatizer().lemmatize(token, tag)

kws = Doc.objects.filter(
    query=q,
    kw__text__iregex='\W'
).values('kw__text').annotate(
    n = Count('pk')
).filter(n__gt=100).order_by('-n')

kw_text = set([x['kw__text'].replace('-',' ') for x in kws])
kw_ws = set([x['kw__text'].replace('-',' ').split()[0] for x in kws]) - stopwords

def tokenize(X):
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

all_ys = range(0,100000)

if "A" == "A":
    X = []
    vecs = []
    ars = AR.objects.filter(ar__gt=0).order_by('ar')
    for ar in ars:
        c_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=stoplist, tokenizer=tokenize)
        abstracts = []
        ys = range(ar.start,ar.end+1)
        ydocs = Doc.objects.filter(query=q,PY__in=ys)
        abdocs = ydocs.filter(content__iregex='\w')
        tidocs = ydocs.exclude(content__iregex='\w')
        abstexts, x, y, z = tutils.proc_docs(abdocs,None)
        abstracts = abstexts + list(tidocs.values_list('title',flat=True))
        X_y = c_vectorizer.fit_transform(abstracts)
        vecs.append(c_vectorizer)
        X.append(X_y)
        print(X_y.shape)

    with open("tables/sizes_X.pickle", "wb") as f:
        pickle.dump(X,f)

    with open("tables/vecs.pickle", "wb") as f:
        pickle.dump(vecs,f)

all_ys = range(0,100000)

f_X = []
f_vecs = []
ars = AR.objects.filter(ar__gt=0).order_by('ar')
for ar in ars:
    c_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=stoplist, tokenizer=fancy_tokenize)
    abstracts = []
    ys = range(ar.start,ar.end+1)
    ydocs = Doc.objects.filter(query=q,PY__in=ys)
    abdocs = ydocs.filter(content__iregex='\w')
    tidocs = ydocs.exclude(content__iregex='\w')
    abstexts, x, y, z = tutils.proc_docs(abdocs,None)
    abstracts = abstexts + list(tidocs.values_list('title',flat=True))
    X_y = c_vectorizer.fit_transform(abstracts)
    f_vecs.append(c_vectorizer)
    f_X.append(X_y)
    print(X_y.shape)

with open("tables/sizes_f_X.pickle", "wb") as f:
    pickle.dump(f_X,f)

with open("tables/f_vecs.pickle", "wb") as f:
    pickle.dump(f_vecs,f)
