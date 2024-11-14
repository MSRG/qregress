#!/bin/bash
dirs=$(find . -name "A2_HWE-CNOT")
topdir=$(pwd)
for i in $dirs; do
  echo $i
  cd $i
  echo $(pwd)
  python3 ../main.py --save_path ./ --settings ./A2_HWE-CNOT.json --train_set ../5_DDCC_train.bin --test_set ../5_DDCC_test.bin --scaler ../5_DDCC_scaler.bin --resume_file partial_state_model.bin
  cd $topdir
  echo $(pwd)
done
