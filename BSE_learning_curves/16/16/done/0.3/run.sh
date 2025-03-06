#! /bin/bash
cwd=$(pwd)
errorname=$(basename `pwd`)
echo $cwd
i="A2_Hadamard/"
name=${i%/}
# Extracting the parent directory name
settings_folder=${name#M-A1-CNOT_Efficient-CRX_}
path="${cwd}/${name}"
export OMP_NUM_THREADS=64
cd $(pwd)/$name
python ${cwd}/main.py --save_path ${path}  --settings ${path}/${name}.json --train_set ${cwd}/PCA16_0.3_Morgan_train.bin --test_set ${cwd}/PCA16_0.3_Morgan_test.bin --scaler ${cwd}/PCA16_0.3_Morgan_scaler.bin  > ${name}.out 2>&1
cd ..
touch ${name}.done

