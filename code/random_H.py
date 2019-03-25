import os, sys, time, resource, re, gc, shutil
from multiprocess import Pool
from functools import partial
from urllib.parse import urlparse, parse_qsl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mongoengine.queryset.visitor import Q
import django
from django.db.models import *
sys.path.append('/home/galm/software/django/tmv/BasicBrowser/')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BasicBrowser.settings")
django.setup()

from scoping.models import *
from tmv_app.models import *
from utils.text import *
from sklearn.feature_extraction.text import *
from sklearn.decomposition import NMF
from scipy.interpolate import make_interp_spline, BSpline

import argparse

parser = argparse.ArgumentParser(description='Show random topic entropy over time')

parser.add_argument('run_id',type=int)
parser.add_argument('fout',type=str)
args = parser.parse_args()

run_id = args.run_id

stat = RunStats.objects.get(pk=run_id)

print(stat.method)

def plot_entropy_time(ax,df,cmap=None,smooth=False):
    lyear = df['py'].max()
    pdf = df[df['py']<=lyear]
    pdf['cat'] = pdf['cat'].astype("category")
    cats = list(pdf[pdf['py']==lyear].sort_values('H',ascending=False)['cat'])

    pdf['cat'].cat.set_categories(cats,inplace=True)
    pdf = pdf.sort_values('cat',ascending=True).reset_index(drop=True)

    for name, group in pdf.groupby('cat'):
        group = group.sort_values('py').reset_index(drop=True)

        x = group['py']
        y = group['H']

        if smooth:
            xnew = np.linspace(x.min(),x.max(),smooth) #300 represents number of points to make between T.min and T.max
            spl = make_interp_spline(x, y, k=3) #BSpline object
            y = spl(xnew)
            x = xnew

        if cmap is not None:
            ax.plot(x,y,label=name,color=cmap[name])
        else:
            ax.plot(x,y,label=name)

    ax.legend()

if stat.method=="DT":
    DTO = DocDynamicTopic.objects
else:
    DTO = DocTopic.objects

import random

Hs = []
pys = list(range(1990,2019))

doc_ids = set(DTO.filter(
    run_id=run_id,
    score__gt=stat.dt_threshold
).values_list('doc_id',flat=True))

for i in [1,2]:
    print(i)
    dids = random.sample(doc_ids,3000)
    dtos = DTO.filter(
        run_id=run_id,
        score__gt=stat.dt_threshold,
        doc__id__in=dids
    )
    for y in pys:
        dts = dtos.filter(doc__PY=y)
        if dts.count()==0:
            continue

        H = 0

        ts = dts.values('topic').annotate(
            pzc = Sum('score')
        )

        for t in ts:
            H+=t['pzc']*np.log(t['pzc'])
        H = -1*H

        Hs.append({
            "cat": i,
            "H": H,
            "py": y,
        })

rdf = pd.DataFrame.from_dict(Hs)

fig, ax = plt.subplots(figsize=(12,12))
plot_entropy_time(ax,rdf, smooth=50)

plt.savefig(f'{args.fout}_{run_id}.png')
