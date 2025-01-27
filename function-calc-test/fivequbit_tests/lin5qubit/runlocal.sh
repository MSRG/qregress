#!/bin/bash
cwd=$(pwd)
errorname=$(basename `pwd`)
echo $cwd
for i in */; do
    name=${i%/}
#   echo "Running $name"
    # Extracting the parent directory name
    settings_folder=${name#M-A1-CNOT_Efficient-CRX_}
    path="${cwd}/${name}"
    number=$(python count_lines.py --path $path/model_log.csv)

    if [ "$number" -ne "1000" ] ;then
	echo "RUN $name: $number"
        cd $(pwd)/$name
        python ${cwd}/main.py --save_path ${path}  --settings ${path}/${name}.json --train_set ${cwd}/linear_train.bin --test_set ${cwd}/linear_test.bin --scaler ${cwd}/linear_scaler.bin --save_circuits True
        
        cd ..
        touch ${name}.done
    fi
done
