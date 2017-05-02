import os, sys, time, resource, re, gc, shutil
from multiprocess import Pool
from functools import partial
from mongoengine import *
from urllib.parse import urlparse, parse_qsl
connect('mongoengine_documents')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mongoengine.queryset.visitor import Q
import django
sys.path.append('/home/galm/software/tmv/BasicBrowser/')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BasicBrowser.settings")
django.setup()
from monary import Monary
mon = Monary()

from scoping.models import *

class scopus_doc(DynamicDocument):
    scopus_id = StringField(required=True, max_length=50, unique=True)
    PY = IntField(required=True)

class scopus_ref(Document):
    text = StringField(required=True, unique=True)
    ti = StringField()
    PY = IntField()
    extra = StringField()
    doi = StringField()
    url = URLField()

class similarity(Document):
    wos_ut = StringField(required=True, max_length=50)
    scopus_id = StringField(required=True, max_length=50,unique_with='wos_ut')
    scopus_do = BooleanField(required=True, max_length=50)
    wos_do = BooleanField(required=True, max_length=50)
    do_match = BooleanField()
    t_match = BooleanField()
    jaccard = FloatField()
    py_diff = IntField()
    wc_diff = IntField()
    wc = IntField()

class match(Document):
    scopus_id = StringField(required=True, max_length=50,unique_with='wos_ut')
    wos_ut = StringField(required=True, max_length=50)
    py_diff = IntField()
    jaccard = FloatField()
    wc_diff = IntField()

def get(r, k):
    try:
        x = r[k]
    except:
        x = ""
    return(x)

def add_doc(r):
    scopus2WoSFields = {
        'TY': 'dt',
        'TI': 'ti',
        'T2': '',
        'C3': '',
        'J2': 'so',
        'VL': 'vl',
        'IS': '',
        'SP': 'bp',
        'EP': 'ep',
        'PY': 'py',
        'DO': 'di',
        'SN': 'sn',
        'AU': 'au',
        'AD': 'ad',
        'AB': 'ab',
        'KW': 'kwp',
        'Y2': '',
        'CY': '',
        #N1 means we need to read the next bit as key
        'Correspondence Address': '',
        'References': '',
        'UR': 'UT', # use url as ut, that's the only unique identifier...
        'PB': ''
        #'ER': , #End record

    }
    django.db.connections.close_all()
    try:
        r['UT'] = r.scopus_id
    except:
        print(r)
        return

    doc = Doc(UT=r['UT'])
    #print(doc)
    doc.title=get(r,'TI')
    doc.content=get(r,'AB')
    doc.PY=get(r,'PY')
    doc.save()
    doc.query.add(q)
    doc.save()
    article = WoSArticle(doc=doc)


    for field in r:
        try:
            f = scopus2WoSFields[field]
            #article.f = r[field]
            setattr(article,f,r[field])
            #article.save()
            #print(r[field])
        except:
            pass

    try:
        article.save()

    except:
        pass


    ## Add authors
    try:
        dais = []
        for a in range(len(r['AU'])):
            #af = r['AF'][a]
            au = r['AU'][a]
            dai = DocAuthInst(doc=doc)
            dai.AU = au
            dai.position = a
            dais.append(dai)
            #dai.save()
        DocAuthInst.objects.bulk_create(dais)
    except:
        print("couldn't add authors")


global q
q = Query.objects.get(pk=354)

for d in scopus_doc.objects.all():
    m = similarity.objects.filter(scopus_id=d.scopus_id)
    t = m.filter(do_match=True)
    if t.count() == 0:
        t = m.filter(jaccard__gt=0.47).order_by('-jaccard')
    if t.count() > 0:
        wosdoc = Doc.objects.get(pk=t.first().wos_ut)
        wosdoc.query.add(q)
        wosdoc.scopus=True
        wosdoc.save()
    else:
        add_doc(d)
