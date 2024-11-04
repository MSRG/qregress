#!/bin/bash
dirs=$(find . -mindepth 1 -maxdepth 1 -type d | sed -e 's,^\./,,')
topdir=$(pwd)
for i in $dirs; do
  if [[ $i != "5qubithelp" ]]; then
   echo "$topdir/$i"
   cd $topdir/$i/M-M-CZ_HWE-CNOT/
   python3 $topdir/$i/main.py --settings $topdir/$i/M-M-CZ_HWE-CNOT/M-M-CZ_HWE-CNOT.json --train_set $topdir/$i/PCA5_0.8_Morgan_train.bin --test_set $topdir/$i/PCA5_0.8_Morgan_test.bin --scaler $topdir/$i/PCA5_0.8_Morgan_scaler.bin --save_path $topdir/$i/M-M-CZ_HWE-CNOT 
   cd $topdir 
  fi
done
