#!/bin/bash
export OMP_NUM_THREADS=64
python3 main.py --settings IQP_Full-Pauli-CRX/IQP_Full-Pauli-CRX.json --train_set quadratic_train.bin --test_set quadratic_test.bin --scaler quadratic_scaler.bin --save_path ./ --resume_file final_state_model.bin  
