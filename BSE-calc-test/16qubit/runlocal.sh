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
 	python3 ${cwd}/main.py --save_path ${path}  --settings ${path}/${name}.json --train_set ${cwd}/PCA16_Morgan_train.bin --test_set ${cwd}/PCA16_Morgan_test.bin --scaler ${cwd}/PCA16_Morgan_scaler.bin >> ${name}.out
	cd ..
    fi
    echo $(pwd)
done
