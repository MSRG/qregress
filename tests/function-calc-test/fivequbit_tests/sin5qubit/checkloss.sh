#!/bin/bash

for i in *.done; do
 extension=${i%.done}
 echo $i
 if [ -f "${extension}/${extension}_results.json" ]; then
   python3 view_loss.py --path ${extension}/model_log.csv
 fi
done
