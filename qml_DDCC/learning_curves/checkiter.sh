#!/bin/bash

files=$(find . -name "model_log.csv" -newermt "$(date +%Y)-$(date +%m)-01")
for i in $files; do
    echo $i
    echo $(awk -F',' '{print $2}' $i | grep -Eo '[0-9]+' | tail -n 1)
done

