
features=(10000 50000 100000)
ngrams=(1 2)

#python3 /home/galm/software/django/tmv/BasicBrowser/manage.py nmf --alpha 0.1 365 100

python dynamic_nmf0.py 3769 80

python dynamic_nmf0.py 3769 100

python dynamic_nmf0.py 3769 120

#python3 /home/galm/software/django/tmv/BasicBrowser/manage.py nmf --alpha 0.05 365 120

#python3 /home/galm/software/django/tmv/BasicBrowser/manage.py nmf --alpha 0.05 365 80

  # for f in "${features[@]}"
  # do
  #   :
  #   for n in "${ngrams[@]}"
  #   do
  #     :
  #     python3 /home/galm/projects/cc-topography/code/nmf_multi.py 20 $n $f
  #   done
  # done
