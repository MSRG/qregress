#!/bin/bash
dir=$(find . -mindepth 1 -maxdepth 1 -type d)
for i in $dir; do
 extension=${i#./}
 echo $extension
 if [ -f "${extension}/${extension}_results.json" ]; then
   python3 view_loss.py --path ${extension}/model_log.csv
 fi
done  
