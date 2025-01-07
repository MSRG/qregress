#!/bin/bash
dir=$(find . -name "M-M-CZ_HWE-CNOT")

for i in $dir; do
 topdir=$(dirname $i)
 echo $topdir
 echo $i
 mv -f $i $topdir/A2_Hadamard
done
