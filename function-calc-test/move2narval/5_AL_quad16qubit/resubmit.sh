#!/bin/bash
rm *done *.o* *.e* *sub
bash run.sh
rm __pycache__.sub quantum.sub
find . -name "*sub" -exec sbatch {} \;
