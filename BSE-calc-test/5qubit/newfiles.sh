#!/bin/bash
newfiles=$(find . -mindepth 1 -maxdepth 1 -type d | sort)

count=0
for i in $newfiles; do
   base=$(basename "$i")
   if [ -f "$i/${base}.json" ]; then
       echo "$base"
       ((count++))   # Increment count correctly
       # rm -rf $base
       # mv -f $i ./
   fi
done

echo "$count"

