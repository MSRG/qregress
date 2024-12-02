#!/bin/bash
csvs=$(find . -name "model_log.csv")

for i in $csvs; do
 sub=$(dirname "$i")
 mv $i $sub/2_model_log.csv
done
