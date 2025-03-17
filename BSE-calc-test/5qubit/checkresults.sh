#!/bin/bash

for i in ./*; do
 extension=${i//.\/}
 if [ -f "${extension}/${extension}_results.json" ]; then
   echo $extension
   results=$(cat "${extension}/${extension}_results.json")
   echo $results
   echo
 fi
done
