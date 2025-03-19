#!/bin/bash

for i in *.done; do
 extension=${i%.done}
 echo $i
 if [ -f "${extension}/${extension}_results.json" ]; then
   results=$(cat "${extension}/${extension}_results.json")
   echo $results
 fi
done
