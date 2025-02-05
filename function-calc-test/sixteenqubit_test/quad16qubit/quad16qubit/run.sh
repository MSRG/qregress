#! /bin/bash

for i in *Full-CRX/; do
  
    name=${i%/}
    topdir=$(pwd) 
    # Extracting the parent directory name
    settings_folder=${name#M-A1-CNOT_Efficient-CRX_}
    path="${topdir}/${name}"
    if [ ! -f "${path}/partial_state_model.bin" ]; then 
      echo $path
      cd $i
      python ${topdir}/main.py --save_path ${path}  --settings ${path}/${name}.json --train_set ${topdir}/quadratic_train.bin --test_set ${topdir}/quadratic_test.bin --scaler ${topdir}/quadratic_scaler.bin --save_circuits True --resume_file ${path}/final_state_model.bin > ${path}/${name}.out 2>&1 
      cd ../
    else
      echo $path
      cd $i
      python ${topdir}/main.py --save_path ${path}  --settings ${path}/${name}.json --train_set ${topdir}/quadratic_train.bin --test_set ${topdir}/quadratic_test.bin --scaler ${topdir}/quadratic_scaler.bin --save_circuits True --resume_file ${path}/partial_state_model.bin > ${path}/${name}.out 2>&1 
      cd ../

    fi
done
#python3 ~/qregress/main.py --settings M_Modified-Pauli-CRZ.json --train_set ~/qregress/function-calc-test/quadratic/quadratic_train.bin --test_set ~/qregress/function-calc-test/quadratic/quadratic_test.bin --scaler ~/qregress/function-calc-test/quadratic/quadratic_scaler.bin
