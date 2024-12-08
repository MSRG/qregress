#!/bin/bash
path="/home/grierjones/qregress/qml_DDCC/RUD_AL_5/RUD_AL/5AL"
python3 ${path}/main.py --settings ${path}/A2_HWE-CNOT/A2_HWE-CNOT.json --train_set ${path}/5_DDCC_train.bin --test_set ${path}/5_DDCC_test.bin --scaler ${path}/5_DDCC_scaler.bin --save_path ./ --save_circuits True 
