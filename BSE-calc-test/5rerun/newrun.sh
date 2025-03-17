#! /bin/bash

# Get the current year and month
current_year=$(date +%Y)
current_month=$(date +%m)

# Find directories to process
dirs=$(find . -mindepth 1 -maxdepth 1 -type d)
for i in ./M-A2-CZ_HWE-CZ; do
#for i in $dirs; do
    name=${i//.\//}
    topdir=$(pwd)
    settings_folder=${name#M-A1-CNOT_Efficient-CRX_}
    path="${topdir}/${name}"

#   echo $path
    # Check for model_log.csv modified in the current month
#   if find "$path" -maxdepth 1 -name "model_log.csv" -newermt "$(date +%Y)-$(date +%m)-01" ! -newermt "$(date +%Y)-$(date +%m)-06" | grep -q .; then
#       echo "Skipping $path because model_log.csv was modified this month."
#       continue
#   fi


    echo $path
    cd $i

    if [ ! -f "${path}/partial_state_model.bin" ]; then
        python ${topdir}/main.py --save_path ${path} --settings ${path}/${name}.json \
        --train_set ${topdir}/PCA5_0.8_Morgan_train.bin --test_set ${topdir}/PCA5_0.8_Morgan_test.bin \
        --scaler ${topdir}/PCA5_0.8_Morgan_scaler.bin --save_circuits True \
        --resume_file ${path}/final_state_model.bin > ${path}/${name}.out 2>&1
    else
        python ${topdir}/main.py --save_path ${path} --settings ${path}/${name}.json \
        --train_set ${topdir}/PCA5_0.8_Morgan_train.bin --test_set ${topdir}/PCA5_0.8_Morgan_test.bin \
        --scaler ${topdir}/PCA5_0.8_Morgan_scaler.bin --save_circuits True \
        --resume_file ${path}/partial_state_model.bin > ${path}/${name}.out 2>&1
    fi

    cd ../
done

