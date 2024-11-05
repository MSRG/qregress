#!/bin/bash
# Grab iteration informations
dir=$(find . -maxdepth 2 -name "partial*" )
#month="Sep"
#month=$(date +%b)

echo $dir

# Loop over directories
for i in $dir; do
 path=$(dirname $i)
 echo $path
 sumiter=0
 # Loop over existing model_log.csv files to find the final iterations
 for j in ${path}/*model_log.csv; do
   echo $j
   raniter=$(cat $j  | grep -iE "Oct" | tail -n 1  | cut -d',' -f2)
   sumiter=$((sumiter + raniter + 1)) 
   echo "$j has $raniter iterations"
   echo $sumiter
 done
 # Write the remaining iterations to file
 remain=$((1000 - sumiter))
 echo "Iterations left: $remain"
 mv ${path}/model_log.csv ${path}/0_model_log.csv
 sed -i 's/"MAX_ITER": 1000/"MAX_ITER": '$remain'/g' ${path}/*.json
 echo $(cat ${path}/*.json)
 echo 
done

rm *done *.o* *.e* *sub
bash run.sh
rm __pycache__.sub quantum.sub helperfiles.sub finished.sub
find . -name "*sub" -exec sbatch {} \;
