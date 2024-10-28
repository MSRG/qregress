#!/bin/bash
python3 main.py --settings M_ESU2/M_ESU2.json --train_set PCA5_0.8_Morgan_train.bin --test_set PCA5_0.8_Morgan_test.bin --scaler PCA5_0.8_Morgan_scaler.bin --save_path ./ --save_circuits True 
