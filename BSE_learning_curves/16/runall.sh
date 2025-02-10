#!/bin/bash
dirs=$(find .  -maxdepth 1 -mindepth 1 -type d | sort)

for i in $dirs; do
 if [[ "$i" != "./done" ]]; then
  echo $i
  cd $i
  bash run.sh
  cd ../
 fi
done
