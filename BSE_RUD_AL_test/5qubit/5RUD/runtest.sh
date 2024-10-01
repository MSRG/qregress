#!/bin/bash
cwd=$(pwd)
name="A2_HWE-CNOT"
path=${cwd}/$name
/opt/miniconda/bin/python ${cwd}/main.py --save_path ${path}  --settings ${path}/${name}.json --train_set ${cwd}/PCA5_Morgan_train.bin --test_set ${cwd}/PCA5_Morgan_test.bin --scaler ${cwd}/PCA5_Morgan_scaler.bin 
