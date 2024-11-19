#!/bin/bash
export OMP_NUM_THREADS=16
cd /home/grierjones/5qubit/A2-A2-CNOT_Efficient-CRZ
python3 /home/grierjones/5qubit/main.py --settings /home/grierjones/5qubit/A2-A2-CNOT_Efficient-CRZ/A2-A2-CNOT_Efficient-CRZ.json --train_set /home/grierjones/5qubit/PCA5_0.8_Morgan_train.bin --test_set /home/grierjones/5qubit/PCA5_0.8_Morgan_test.bin --scaler /home/grierjones/5qubit/PCA5_0.8_Morgan_scaler.bin --save_path /home/grierjones/5qubit/A2-A2-CNOT_Efficient-CRZ --save_circuits True --resume_file /home/grierjones/5qubit/A2-A2-CNOT_Efficient-CRZ/partial_state_model.bin 
cd /home/grierjones/5qubit/
