#!/bin/bash
export OMP_NUM_THREADS=64
python3 main.py --settings A2_HWE-CNOT/A2_HWE-CNOT.json --train_set PCA5_0.8_Morgan_train.bin --test_set PCA5_Morgan_test.bin --scaler PCA5_Morgan_scaler.bin --save_path ./
