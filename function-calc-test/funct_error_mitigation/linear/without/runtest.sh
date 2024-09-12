#!/bin/bash
python3 main.py --settings IQP_Full-Pauli-CRZ/IQP_Full-Pauli-CRZ.json --train_set linear_train.bin --test_set linear_test.bin --scaler linear_scaler.bin --save_path ./IQP_Full-Pauli-CRZ --save_circuits True 
