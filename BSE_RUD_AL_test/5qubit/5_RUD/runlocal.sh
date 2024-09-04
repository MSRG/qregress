#! /bin/bash
cwd=$(pwd)
for i in */; do
    name=${i%/}
    # Extracting the parent directory name
    settings_folder=${name#M-A1-CNOT_Efficient-CRX_}
    path="${cwd}/${name}"
    if [ ! -f ${path}/${name}_results.json ] && [ -f ${path}/${name}.json ]; then
    	echo $path
        echo "${name}.done not found!"
 	cd $path 

	if [ -f ${path}/partial_state_model.bin ]; then
	     python3 ${cwd}/main.py --save_path ${path}  --settings ${path}/${name}.json --train_set ${cwd}/PCA5_Morgan_train.bin --test_set ${cwd}/PCA5_Morgan_test.bin --scaler ${cwd}/PCA5_Morgan_scaler.bin --resume_file ${path}/partial_state_model.bin > ${name}.out
	else
	     python3 ${cwd}/main.py --save_path ${path}  --settings ${path}/${name}.json --train_set ${cwd}/PCA5_Morgan_train.bin --test_set ${cwd}/PCA5_Morgan_test.bin --scaler ${cwd}/PCA5_Morgan_scaler.bin  > ${name}.out
	fi
	cd ..
    fi
    echo $(pwd)
done
