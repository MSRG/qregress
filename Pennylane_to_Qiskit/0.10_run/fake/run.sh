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
  cat > ${dirname}.sub <<EOF
#!/bin/bash
#SBATCH -t 4-00:00:00
#SBATCH -J ${dirname} 
#SBATCH -N 1
#SBATCH --cpus-per-task 64
#SBATCH --account=rrg-jacobsen-ab
#SBATCH --error=A2_HWE-CNOT.e%J
#SBATCH --output=A2_HWE-CNOT.o%J

export OMP_NUM_THREADS=64


/lustre06/project/6006115/gjones/env/bin/python3 fake.py > fake.out 2>&1

EOF
	cd ../
 done
done
