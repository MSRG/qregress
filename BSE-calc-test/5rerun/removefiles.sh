#!/bin/bash

paths=$(grep -i '"MAX_ITER": 1000' */*json | awk '{print $1}' | xargs -n 1 dirname)


for i in $paths; do
  echo "${i}"
  rm $i/*bin $i/*csv */*svg */*_results.json
  ls $i
done


