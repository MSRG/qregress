#!/bin/bash
# Grab iteration informations
dir=$(find . -maxdepth 3 -name "partial*" )
#month="Sep"
#month=$(date +%b)

topdir=$(pwd)

# Loop over directories
for i in $dir; do
 echo "Top directory: $topdir"
 path=$(dirname $i)
 echo "Data directory: $path"

## Loop over existing model_log.csv files to find the final iterations
#sumiter=0
#for j in ${path}/*model_log.csv; do
#  echo $j
#  raniter=$(cat $j  | grep -iE "Nov" | tail -n 1  | cut -d',' -f2)
#  sumiter=$((sumiter + raniter + 1)) 
#  echo "$j has $raniter iterations"
#  echo $sumiter
#done
 
## Write the remaining iterations to file
#remain=$((1000 - sumiter))
#echo "Iterations left: $remain"
 mv ${path}/model_log.csv ${path}/10_model_log.csv
#sed -i 's/"MAX_ITER": 500/"MAX_ITER": '$remain'/g' ${path}/*.json
 
 # Go into directory and submit
 cd $(dirname "$path")
 cp ../run.sh .
 echo "cd to $(pwd)"
 rm *done *.o* *.e* *sub
 bash run.sh
 rm __pycache__.sub quantum.sub helperfiles.sub finished.sub
 find . -name "*sub" -exec sbatch {} \;
 echo $(cat */*.json)
 
 # Move back to top
 cd $topdir
 echo $(pwd)
 echo 
done


