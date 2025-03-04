#!/bin/bash
#optimization_level = 3
#shots = 1024.0
#resilience_level = 1

for ol in 0;do
 for rl in {0..2};do
  dirname="ol${ol}_rl${rl}"
  if [ ! -d ${dirname} ] ; then
    echo "${dirname} does not exist"
    mkdir ${dirname}
  else
    echo "${dirname} does exist"
  fi
  papermill convert_ML_batched.ipynb ${dirname}/convert_ML_batched.ipynb -r optimization_level ${ol} -r resilience_level ${rl}
  mv ./model_log.csv ./final_state_model.bin ./A2_HWE-CNOT_plot.svg ./A2_HWE-CNOT_results.json ./A2_HWE-CNOT_predicted_values.csv ${dirname}/ 
 done
done
