#!/bin/bash

for i in *Full-CRX/; do
 extension=${i%/}
 echo $i
 if [ -f "${extension}/${extension}_results.json" ]; then
  python3 view_loss.py --path ${extension}/model_log.csv
 fi
done
