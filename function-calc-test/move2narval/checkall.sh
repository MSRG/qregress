#!/bin/bash
dirs=$(find . -maxdepth 1 -type d)
for i in $dirs; do
  echo $i
  if [ -e "$i/run.sh" ]; then
      echo "File exists"
      cd $i
      echo "$(find . -name "*results.json" | wc -l)"
      cd ../
  fi 
done
