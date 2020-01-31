import sys, os, django

sys.path.append('/home/galm/software/django/tmv/BasicBrowser/')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BasicBrowser.settings")
django.setup()

from scoping.models import *
from tmv_app.models import *
from tmv_app.tasks import *

sw = set([])

sw.add('elsevier')
sw.add('rights')
sw.add('reserved')
sw.add('john')
sw.add('wiley')
sw.add('sons')
sw.add('copyright')

q = Query.objects.get(pk=6187)
for m in ["NM","LD"]:
    for a in [0.01,0.05,0.1]:
        for k in [80,90,100,110,120,130,140,150]:
            if m=="LD":
                alpha=a*10
            else:
                alpha=a
            try:
                stat, created = RunStats.objects.get_or_create(
                    K=k,
                    alpha=alpha,
                    fancy_tokenization=True,
                    max_df=0.9,
                    max_iter=500,
                    method=m,
                    min_freq=50,
                    ngram=1,
                    query=q,
                )
            except:
                continue

            if created or stat.status==0:
                stat.extra_stopwords=list(sw)
                stat.save()
                do_nmf(stat.pk)
