#! /bin/bash
cwd=$(pwd)
errorname=$(basename `pwd`)
echo $cwd
for i in ./A2_HWE-CNOT; do
    name="${i#./}"
    echo $name
    # Extracting the parent directory name
    settings_folder=${name#M-A1-CNOT_Efficient-CRX_}
    path="${cwd}/${name}"
    if [ ! -f ${path}/${name}_results.json ]; then
        echo "${name}.done not found!"
        export OMP_NUM_THREADS=80
        cd $(pwd)/$name
        python ${cwd}/main.py --save_path ${path}  --settings ${path}/${name}.json --train_set ${cwd}/5_DDCC_train.bin --test_set ${cwd}/5_DDCC_test.bin --scaler ${cwd}/5_DDCC_scaler.bin > ${name}.out 2>&1
        
        cd ..
        touch ${name}.done

    fi
    echo "Done ${name}.sub"
done
