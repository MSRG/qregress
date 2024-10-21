#!/bin/bash
dirs=$(find . -mindepth 1 -maxdepth 1 -type d)
for i in $dirs; do
  echo $i
  cd $i
  echo $(pwd)
  cp ../run.sh .
  rm *done *.o* *.e* *sub
  bash run.sh
  rm __pycache__.sub quantum.sub helperfiles.sub
  find . -name "*sub" -exec sbatch {} \;
  cd ../
done
