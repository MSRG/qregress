#!/bin/bash
for i in 0.5 0.7 0.8; do
  echo $i
  cd $i
  echo $(pwd)
  echo "File exists $i/run.sh"
  cp -r helperfiles/quantum .
  cp -r helperfiles/*py .
  bash runtest.sh
  cd ../
done
