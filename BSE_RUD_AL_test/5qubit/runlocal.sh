#! /bin/bash
cwd=$(pwd)
errorname=$(basename `pwd`)
echo $cwd
for i in $(find . -maxdepth 2 -name "*.json" -exec dirname {} \;); do
    name=${i%/}
    # Extracting the parent directory name
    settings_folder=${name#M-A1-CNOT_Efficient-CRX_}
    path="${cwd}/${name}"
    if [ ! -f "${path}/${name}_results.json" ]; then
	export OMP_NUM_THREADS=16
	cd ${path}
        echo $(pwd)
        echo ${cwd}
	echo ${path}
	echo
        python3 /home/grierjones/5qubit/main.py --settings /home/grierjones/5qubit/${name}/${name}.json --train_set /home/grierjones/5qubit/PCA5_0.8_Morgan_train.bin --test_set /home/grierjones/5qubit/PCA5_0.8_Morgan_test.bin --scaler /home/grierjones/5qubit/PCA5_0.8_Morgan_scaler.bin --save_path /home/grierjones/5qubit/${name} --save_circuits True 
	
	cd ..
    fi
done
