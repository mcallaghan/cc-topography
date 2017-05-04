topics=(20 40 75 100)
features=(5000 10000 50000)
ngrams=(1 2)


for t in "${topics[@]}"
do
    :
    for f in "${features[@]}"
    do
      :
      for n in "${ngrams[@]}"
      do
        :
        python3 /home/galm/projects/cc-topography/code/nmf.py $t $n $f
      done
    done
done
