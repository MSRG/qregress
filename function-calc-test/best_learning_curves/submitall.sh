#!/bin/bash
dirs=$(find . -maxdepth 1 -type d)
for i in $dirs; do
  echo $i
  if [ -e "$i/run.sh" ]; then
      echo "File exists"
      cd $i
      rm *done *.o* *.e* *sub
      bash run.sh
      unzip helperfiles.zip
      cp -r helperfiles/* .
      rm quantum.sub __pycache__.sub helperfiles.sub 
      find . -name "*sub" -exec sbatch {} \;
      cd ../
  fi 
done
