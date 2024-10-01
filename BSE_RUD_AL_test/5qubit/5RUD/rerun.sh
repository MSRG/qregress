#!/bin/bash
# Grab iteration informations
month=$(date +%b)
raniter=$(cat */model_log.csv  | grep -iE "$month" | tail -n 1  | cut -d',' -f2)
remain=$((1000-$raniter))
echo "Iterations left: $remain"

# Change iterations
sed -i 's/"MAX_ITER": 1000/"MAX_ITER": '$remain'/g' */*.json
cat */*.json

# run resubmit.sh
rm *done *.o* *.e* *sub
bash run.sh
rm __pycache__.sub quantum.sub helperfiles.sub
find . -name "*sub" -exec sbatch {} \;
