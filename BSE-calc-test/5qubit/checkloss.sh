#!/bin/bash

for i in *; do
 extension=${i}
 if [ -f "${extension}/${extension}_results.json" ]; then
   echo $i
   python3 view_combined.py --path ${extension}
 fi
done
