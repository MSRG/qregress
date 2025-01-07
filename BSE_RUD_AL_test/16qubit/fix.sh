#!/bin/bash
for i in ./* ; do
 if [[ $i != '5qubithelp' && -d $i ]]; then
 echo $i
 cd $i 
 cp ../add_params.py ./ 
 python3 add_params.py 
 cd ../
 fi
done
