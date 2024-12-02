#!/bin/bash
for i in 0.1 0.3 0.5 0.7 0.8; do
  echo $i
  cd $i
  echo $(pwd)
  rm *done *.o* *.e* *sub
  bash run.sh
  rm __pycache__.sub quantum.sub helperfiles.sub
  find . -name "*sub" -exec sbatch {} \;
  cd ../
done
