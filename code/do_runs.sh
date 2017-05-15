
features=(10000 50000 100000)
ngrams=(1 2)


  for f in "${features[@]}"
  do
    :
    for n in "${ngrams[@]}"
    do
      :
      python3 /home/galm/projects/cc-topography/code/nmf_multi.py 20 $n $f
    done
  done
