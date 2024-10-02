#!/bin/bash
# Grab iteration informations
dir=$(find . -maxdepth 2 -name "partial*")
month="Sep"
#month=$(date +%b)



for i in $dir; do
 path=$(dirname $i)
 raniter=$(cat ${path}/model_log.csv  | grep -iE "$month" | tail -n 1  | cut -d',' -f2)
 echo $path
 echo $raniter
 remain=$((1000-$raniter))
 echo "Iterations left: $remain"
 mv ${path}/model_log.csv ${path}/0_model_log.csv
 sed -i 's/"MAX_ITER": 1000/"MAX_ITER": '$remain'/g' ${path}/*.json
 echo $(cat ${path}/*.json)
 echo 

done

## Change iterations

#cat */*.json
#
## run resubmit.sh
#rm *done *.o* *.e* *sub
#bash run.sh
#rm __pycache__.sub quantum.sub helperfiles.sub
#find . -name "*sub" -exec sbatch {} \;
