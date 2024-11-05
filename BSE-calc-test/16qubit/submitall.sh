#!/bin/bash
dirs=$(find . -mindepth 1 -maxdepth 1 -type d | sed -e 's,^\./,,')
topdir=$(pwd)
for i in $dirs; do
  if [[ -f $topdir/$i/$i.json ]]; then
   echo "$topdir/$i"
   cd $topdir/$i
   cp model_log.csv 1_model_log.csv
   python3 $topdir/main.py --settings $topdir/$i/$i.json --train_set $topdir/PCA16_0.8_Morgan_train.bin --test_set $topdir/PCA16_0.8_Morgan_test.bin --scaler $topdir/PCA16_0.8_Morgan_scaler.bin --save_path $topdir/$i --resume_file $topdir/$i/partial_state_model.bin
   cd $topdir 
  fi
done
