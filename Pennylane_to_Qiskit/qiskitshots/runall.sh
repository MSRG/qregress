#!/bin/bash
#optimization_level = 3
#shots = 1024.0
#resilience_level = 1

for i in {1..10};do
  shots=$((i * 1024))
  dirname="shots_${shots}"
  if [ ! -d ${dirname} ] ; then
    echo "${dirname} does not exist"
    mkdir ${dirname}
  else
    echo "${dirname} does exist"
  fi
  papermill convert_ML_batched.ipynb ${dirname}/convert_ML_batched.ipynb -r shots ${shots} 
  mv ./model_log.csv ./final_state_model.bin ./A2_HWE-CNOT_plot.svg ./A2_HWE-CNOT_results.json ./A2_HWE-CNOT_predicted_values.csv ${dirname}/ 
done
