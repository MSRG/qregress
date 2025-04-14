#!/bin/bash

for rl in 0 1 2; do
 for ol in 0 1 2 3; do
	dirname="ol${ol}_rl${rl}"
  if [ ! -d $dirname ]; then
		mkdir $dirname
  fi
  cp *bin $dirname
  cp QiskitRegressor.py $dirname
  cp fake.py $dirname
  cd $dirname

  sed -i "s/optimization_level = 0/optimization_level = ${ol}/g" fake.py
  sed -i "s/resilience_level = 2/resilience_level = ${rl}/g" fake.py
  export OMP_NUM_THREADS=64
  echo "Running $dirname using $OMP_NUM_THREADS" 
  python3 fake.py > fake.out 2>&1
  cd ../
 done
done
