#!/bin/bash
cwd=$(pwd)
errorname=$(basename `pwd`)
echo $cwd
for i in */; do
    name=${i%/}
    echo "Running $name"
    # Extracting the parent directory name
    settings_folder=${name#M-A1-CNOT_Efficient-CRX_}
    path="${cwd}/${name}"
    cd $(pwd)/$name
    
    python ${cwd}/main.py --save_path ${path}  --settings ${path}/${name}.json --train_set ${cwd}/quadratic_train.bin --test_set ${cwd}/quadratic_test.bin --scaler ${cwd}/quadratic_scaler.bin --save_circuits True 
    
    cd ..
    touch ${name}.done

done
