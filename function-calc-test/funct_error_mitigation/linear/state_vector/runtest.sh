#!/bin/bash
export OMP_NUM_THREADS=64
python3 main.py --settings IQP_Full-Pauli-CRZ/IQP_Full-Pauli-CRZ.json --train_set linear_train.bin --test_set linear_test.bin --scaler linear_scaler.bin --save_path ./ --save_circuits True --resume_file final_state_model.bin  
