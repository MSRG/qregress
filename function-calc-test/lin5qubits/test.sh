#! /bin/bash

for i in */; do
    name=${i%/}
    if [ ! -f  ${name}.done ]; then
        # Extracting the parent directory name
        settings_folder=${name#M-A1-CNOT_Efficient-CRX_}

        echo "Not ran ${name}"
    fi
done
#python3 ~/qregress/main.py --settings M_Modified-Pauli-CRZ.json --train_set ~/qregress/function-calc-test/linear/linear_train.bin --test_set ~/qregress/function-calc-test/linear/linear_test.bin --scaler ~/qregress/function-calc-test/linear/linear_scaler.bin
