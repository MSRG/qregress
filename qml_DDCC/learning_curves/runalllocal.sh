#!/bin/bash
cwd=$(pwd)
for dir in 0.1 0.3 0.5 0.7 0.8; do
  echo "Running $dir"
  cd $dir/A2_HWE-CNOT/
  echo "$(pwd)"
  cp -r ../helperfiles/* .
  python3 main.py --settings A2_HWE-CNOT.json --train_set "../${dir}_5_DDCC_train.bin" --test_set "../${dir}_5_DDCC_test.bin" --scaler "../${dir}_5_DDCC_scaler.bin" --save_path $(pwd) --save_circuits True > A2_HWE-CNOT.out 2>&1
  cd $cwd 
done

