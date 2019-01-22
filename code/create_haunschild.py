import django
import sys, os
import pandas as pd
#import matplotlib.pyplot as plt

sys.path.append('/home/galm/software/django/tmv/BasicBrowser/')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BasicBrowser.settings")
django.setup()

from scoping.models import *

q = Query.objects.get(pk=3771)

docs = q.doc_set.all()

# Necessary regular expressions
r1 = "(\w*climat\w* chang\w*)|(\wclimate\w* warming\*)(\w*global temperature\w*)|(\w*global warming\w*)|(\w*greenhouse gas\w*)|(\w*greenhouse effect\w*)|(\w*greenhouse warming\w*)"
r2 = "(\w*climat\w*)"
r3 = "(\w*acclimat\w*)"
r4 = "(\bclimat\w*)"

# Title search
step2_1 = set(docs.filter(title__iregex=r1).values_list('pk',flat=True))

step2_2 = set(docs.filter(title__iregex=r2).values_list('pk',flat=True))

step2_3 = set(docs.filter(pk__in=step2_2,title__iregex=r3).exclude(title__iregex=r4).values_list('pk',flat=True))

step2 = step2_1 | (step2_2 - step2_3)

# Abstracts
step3 = set(docs.filter(content__iregex=r1).values_list('pk',flat=True))

# Keywords
step4_1 = set(docs.filter(wosarticle__de__iregex=r1).values_list('pk',flat=True))

step4_2 = set(docs.filter(wosarticle__de__iregex=r2).values_list('pk',flat=True))

step4_3 = set(docs.filter(
  pk__in=step4_2,
  wosarticle__de__iregex=r3
).exclude(
  wosarticle__de__iregex=r4
).values_list('pk',flat=True))

step4 = step4_1 | (step4_2 - step4_3)

# Keywords plus
step4_1_2 = set(docs.filter(wosarticle__kwp__iregex=r1).values_list('pk',flat=True))

step4_2_2 = set(docs.filter(wosarticle__kwp__iregex=r2).values_list('pk',flat=True))

step4_3_2 = set(docs.filter(
  pk__in=step4_2,
  wosarticle__kwp__iregex=r3
).exclude(
  wosarticle__kwp__iregex=r4
).values_list('pk',flat=True))

step4__2 = step4_1_2 | (step4_2_2 - step4_3_2)

step5 = step2 | step3 | step4 | step4__2

s5docs = Doc.objects.filter(pk__in=step5)

print(s5docs.filter(
    PY__in=list(range(1980,2015)),
    wosarticle__dt__in=["Article","Review"]
).count())

new_q, created = Query.objects.get_or_create(
    title="Haunschild - updated",
    text="Manually filter Haunschild query",
    project=q.project,
    creator=User.objects.get(username="galm")
)

for d in s5docs:
    d.query.add(new_q)
