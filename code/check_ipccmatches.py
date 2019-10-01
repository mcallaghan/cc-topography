
import os, sys, time, resource, re, gc, shutil
import argparse
import pandas as pd

import django

sys.path.append('/home/galm/software/django/tmv/BasicBrowser/')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BasicBrowser.settings")
django.setup()

from scoping.models import *

parser = argparse.ArgumentParser(description='Check unmatched IPCC References')
parser.add_argument('sample_size', type=int)

args = parser.parse_args()

already_rated = IPCCRef.objects.filter(match_status__gt=0)

if already_rated.count() < args.sample_size:

    qdocs = set(IPCCRef.objects.filter(doc__query=6187).values_list('pk',flat=True))

    unmatched = list(IPCCRef.objects.exclude(id__in=qdocs).values_list('pk',flat=True))

    def random_list(l,n):
        return [ l[i] for i in sorted(random.sample(range(len(l)), n)) ]

    already_rated = IPCCRef.objects.filter(pk__in=unmatched, match_status__gt=0)

    docs_left = args.sample_size-already_rated.count()

    print(docs_left)

    if docs_left > 0:

        ir_sample = IPCCRef.objects.filter(id__in=random_list(unmatched, docs_left))

        for i,ir in enumerate(ir_sample):
            if ir.doc:
                ir.match_status=2
                ir.save()
            else:
                print(f"\n{docs_left-i} docs left!")
                print(f"\n{ir.text} - {ir.year} - {ir.authors}")
                ms = input("\nis this 1 - in the query; 2 - in the WoS; 3 - not in WoS: \n")
                ir.match_status=ms
                ir.save()

        already_rated = IPCCRef.objects.filter(pk__in=unmatched, match_status__gt=0)

results = []
for r in already_rated:
    results.append({
        "wgs": "; ".join([wg.__str__() for wg in r.wg.all()]),
        "text":  f"{r.text} - {r.year} - {r.authors}",
        "match_status": r.get_match_status_display()
    })

df = pd.DataFrame.from_dict(
    results
)

df.to_csv("../tables/ipcc_ref_matching.csv", index=False)

for s in IPCCRef.MATCH_STATUS:
    p = df[df['match_status']==s[1]].shape[0] / df.shape[0]
    print(f"{s[1]} - {p:.0%}")


for ir in already_rated.filter(match_status=1):
    print(f"{ir.text} - {ir.year} - {ir.authors}")
    print(ir.doc)
print("finished!")
