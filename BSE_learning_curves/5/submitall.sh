#!/bin/bash
for i in 0.1 0.3 0.5 0.7 0.8; do
  echo $i
  cd $i
  echo $(pwd)
  cp ../runtest.sh .
  bash runtest.sh
  cd ../
done
