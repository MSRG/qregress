#!/bin/bash
export OMP_NUM_THREADS=64
topdir=$(dirname "$(pwd)")
python3 ${topdir}/main.py --settings $(pwd)/IQP_Full-Pauli-CRX.json --train_set ${topdir}/quadratic_train.bin --test_set ${topdir}/quadratic_test.bin --scaler ${topdir}/quadratic_scaler.bin --save_path $(pwd) --resume_file $(pwd)/final_state_model.bin  
