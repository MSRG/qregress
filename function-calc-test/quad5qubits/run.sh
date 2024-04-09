#! /bin/bash

for i in */; do
    name=${i%/}
    # Extracting the parent directory name
    settings_folder=${name#M-A1-CNOT_Efficient-CRX_}
    path="/home/gjones/scratch/quad5qubits/${name}"
    if [ ! -f ${path}/${name}_results.json ]; then
        echo "${name}.done not found!"
    	cat > ${name}.sub <<EOF
#! /bin/bash
#SBATCH -t 7-00:00:00
#SBATCH -J quadratic_${name}
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --account=rrg-jacobsen-ab
#SBATCH --error=${name}.e%J        # The file where run time errors will be dumped
#SBATCH --output=${name}.o%J               # The file where the output of the terminal will be dumped

module load apptainer 
cd $(pwd)/$name

apptainer run -C -B /home/gjones/scratch/quad5qubits ~/deb.sif /opt/miniconda/bin/python /home/gjones/scratch/quad5qubits/main.py --save_path ${path}  --settings ${path}/${name}.json --train_set /home/gjones/scratch/quad5qubits/quadratic_train.bin --test_set /home/gjones/scratch/quad5qubits/quadratic_test.bin --scaler /home/gjones/scratch/quad5qubits/quadratic_scaler.bin >> ${name}.out 2>&1 

cd ..
touch ${name}.done

EOF
    fi
    echo "Done ${name}.sub"
done
