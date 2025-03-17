#!/bin/bash
RED='\033[0;31m'
NC='\033[0m' # No Color
files=$(find . -name "model_log.csv" | sort -h )
for i in $files; do
   #echo $i
    iters=$(awk -F',' '{print $2}' $i | grep -Eo '[0-9]+' | tail -n 1)
    if (( $iters !=  999  )); then
     echo -e "${RED}$i: $iters${NC}"
    fi
done

