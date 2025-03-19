#!/bin/bash
python3 ../main.py --settings $(pwd)/M_Full-CRX.json --train_set ../linear_train.bin --test_set ../linear_test.bin --scaler ../linear_scaler.bin --save_path $(pwd)
