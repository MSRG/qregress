#! /bin/bash

for i in */; do
    name=${i%/}
    
    # Extracting the parent directory name
    settings_folder=${name#M-A1-CNOT_Efficient-CRX_}
    path="/scratch/j/jacobsen/gjones/qregress/function-calc-test/lin5qubits/${name}"
    if [ ! -f ${name}.done ]; then
        echo "${name}.done not found!"
    	cat > ${name}.sub <<EOF
#! /bin/bash
#SBATCH -t 0-23:59:00
#SBATCH -J ${name}
#SBATCH -N 1
#SBATCH -n 80
#SBATCH --account=def-fekl
#SBATCH --error=${name}.e%J        # The file where run time errors will be dumped
#SBATCH --output=${name}.o%J               # The file where the output of the terminal will be dumped

module load apptainer 
cd $(pwd)/$name

apptainer run -C -B  ~/deb.sif /opt/miniconda/bin/python /main.py --save_path ${path}  --settings ${path}/${name}.json --train_set /linear_train.bin --test_set /linear_test.bin --scaler /linear_scaler.bin >> ${name}.out 2>&1 

cd ..
touch ${name}.done

EOF
    fi
    echo "Done ${name}.sub"
done
#python3 ~/qregress/main.py --settings M_Modified-Pauli-CRZ.json --train_set ~/qregress/function-calc-test/linear/linear_train.bin --test_set ~/qregress/function-calc-test/linear/linear_test.bin --scaler ~/qregress/function-calc-test/linear/linear_scaler.bin
