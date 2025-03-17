#!/bin/bash
rm *done *.o* *.e* *sub
python update.py
bash run.sh
rm __pycache__.sub quantum.sub helperfiles.sub
find . -name "*sub" -exec sbatch {} \;
