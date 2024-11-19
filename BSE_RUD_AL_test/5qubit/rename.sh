#!/bin/bash
dir=$(find . -name "A2_HWE-CNOT.json")

for i in $dir; do
 topdir=$(dirname $i)
 echo $topdir
 echo $i
 mv -f $i $topdir/M-M-CZ_HWE-CNOT.json
done
