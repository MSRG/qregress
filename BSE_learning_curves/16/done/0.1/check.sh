#!/bin/bash 
# All directories - running = diff.txt
dir=$(cat diff.txt)

# Find the ones that are done
for i in $dir; do
 #echo $i
 if [ -f ${i}/${i}_results.json ]; then
  echo $i >> done.txt
  echo "$(basename $i)"
  cat ${i}/${i}_results.json
  echo
 fi 
done
