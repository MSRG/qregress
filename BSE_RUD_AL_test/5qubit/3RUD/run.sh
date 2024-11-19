#! /bin/bash
cwd=$(pwd)
errorname=$(basename `pwd`)
echo $cwd
for i in */; do
    name=${i%/}
    # Extracting the parent directory name
    settings_folder=${name#M-A1-CNOT_Efficient-CRX_}
    path="${cwd}/${name}"
    if [ ! -f ${path}/partial* ]; then
        echo "${name}.done not found!"
    	cat > ${name}.sub <<EOF
#! /bin/bash
#SBATCH -t 7-00:00:00
#SBATCH -J ${errorname}_linear_${name}
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --account=rrg-jacobsen-ab
#SBATCH --error=${name}.e%J        # The file where run time errors will be dumped
#SBATCH --output=${name}.o%J               # The file where the output of the terminal will be dumped


export OMP_NUM_THREADS=64
cd $(pwd)/$name
/lustre06/project/6006115/gjones/env/bin/python ${cwd}/main.py --save_path ${path}  --settings ${path}/${name}.json --train_set ${cwd}/PCA5_0.8_Morgan_train.bin --test_set ${cwd}/PCA5_0.8_Morgan_test.bin --scaler ${cwd}/PCA5_0.8_Morgan_scaler.bin > ${name}.out 2>&1

cd ..
touch ${name}.done

EOF
    else
        echo "${name} resume!"
    	cat > ${name}.sub <<EOF
#! /bin/bash
#SBATCH -t 7-00:00:00
#SBATCH -J ${errorname}_linear_${name}
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --account=rrg-jacobsen-ab
#SBATCH --error=${name}.e%J        # The file where run time errors will be dumped
#SBATCH --output=${name}.o%J               # The file where the output of the terminal will be dumped


export OMP_NUM_THREADS=64
cd $(pwd)/$name
/lustre06/project/6006115/gjones/env/bin/python ${cwd}/main.py --save_path ${path}  --settings ${path}/${name}.json --train_set ${cwd}/PCA5_0.8_Morgan_train.bin --test_set ${cwd}/PCA5_0.8_Morgan_test.bin --scaler ${cwd}/PCA5_0.8_Morgan_scaler.bin --resume_file ${path}/partial_state_model.bin > ${name}.out 2>&1

cd ..
touch ${name}.done

EOF

    fi
    echo "Done ${name}.sub"
done
