#!/bin/bash
cp -r helperfiles/* .
rm *done *.o* *.e* *sub
bash run.sh
rm __pycache__.sub quantum.sub helperfiles.sub
find . -name "*sub" -exec sbatch {} \;
