#! /bin/bash

for i in */; do
    name=${i%/}
    
    # Extracting the parent directory name
    settings_folder=${name#M-A1-CNOT_Efficient-CRX_}
    

    cd $(pwd)/$name
    python ~/qregress/main.py --settings ${name}.json --train_set ~/qregress/function-calc-test/linear/linear_train.bin --test_set ~/qregress/function-calc-test/linear/linear_test.bin --scaler ~/qregress/function-calc-test/linear/linear_scaler.bin >> ${name}.out 2>&1
    
    cd ..
    touch ${name}.done

    echo "Done ${name}"
done
#python3 ~/qregress/main.py --settings M_Modified-Pauli-CRZ.json --train_set ~/qregress/function-calc-test/linear/linear_train.bin --test_set ~/qregress/function-calc-test/linear/linear_test.bin --scaler ~/qregress/function-calc-test/linear/linear_scaler.bin
