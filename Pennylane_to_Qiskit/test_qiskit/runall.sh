#!/bin/bash
#optimization_level = 3
#shots = 1024.0
#resilience_level = 1

for ol in {1..3};do
 for rl in {1..2};do
  dirname="ol${ol}_rl${rl}"
  if [ ! -d ${dirname} ] ; then
    echo "${dirname} does not exist"
    mkdir ${dirname}
  else
    echo "${dirname} does exist"
  fi
  papermill convert_ML_batched.ipynb ${dirname}/convert_ML_batched.ipynb -r optimization_level ${ol} -r resilience_level ${rl} 
 done
done
