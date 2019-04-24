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

for k in [100]:
    for a in [0.01,0.05,0.1]:
        for m in ["NM","LD"]:
            if m=="LD":
                alpha=a*10
            else:
                alpha=a
            stat, created = RunStats.objects.get_or_create(
                K=k,
                alpha=alpha,
                fancy_tokenization=True,
                extra_stopwords=list(sw),
                max_df=0.9,
                max_iter=200,
                method=m,
                min_freq=50,
                ngram=1,
                query=q
            )
            do_nmf(stat.pk)