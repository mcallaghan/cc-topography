import sys
import django
import os
# import file for easy access to browser database
sys.path.append('/home/galm/software/django/tmv/BasicBrowser/')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BasicBrowser.settings")
django.setup()

from tmv_app.models import *
from scoping.models import *
from tmv_app.tasks import *

K = int(sys.argv[1])

q = Query.objects.get(pk=2355)

stat = RunStats(
    query=q,
    K=K,
    min_freq=5,
    method='NM',
    citations=True,
    max_iterations=500
)

stat.save()

do_nmf(stat.pk)
