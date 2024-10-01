#!/bin/bash
dirs=$(find . -maxdepth 1 -mindepth 1 -type d)
echo $(pwd)
for i in $dirs; do
  echo $i
  cd $i
  echo $(pwd)
  cp ../run.sh .
  echo "File exists $i/run.sh"
  if [ -e "run.sh" ]; then
      #cp -r helperfiles/* .
      rm *done *.o* *.e* *sub
      bash run.sh
      rm quantum.sub __pycache__.sub helperfiles.sub
      find . -name "*sub" -exec sbatch {} \;
  fi
  cd ../
done
