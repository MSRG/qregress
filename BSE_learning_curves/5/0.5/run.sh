#! /bin/bash
cwd=$(pwd)
errorname=$(basename `pwd`)
echo $cwd
for i in */; do
    name=${i%/}
    # Extracting the parent directory name
    settings_folder=${name#M-A1-CNOT_Efficient-CRX_}
    path="${cwd}/${name}"
    if [ ! -f ${path}/partial_state_model.bin ]; then
        echo "${name}.done not found!"
    	cat > ${name}.sub <<EOF
#! /bin/bash
#SBATCH -t 0-23:59:59
#SBATCH -J ${errorname}_PCA5_0.5_Morgan_${name}
#SBATCH -N 1
#SBATCH -n 80
#SBATCH --account=rrg-fekl-ac
#SBATCH --error=${name}.e%J        # The file where run time errors will be dumped
#SBATCH --output=${name}.o%J               # The file where the output of the terminal will be dumped

cd $(pwd)/$name

/scinet/niagara/software/2019b/opt/base/python/3.11.5/bin/python3 ${cwd}/main.py --save_path ${path}  --settings ${path}/${name}.json --train_set ${cwd}/PCA5_0.5_Morgan_train.bin --test_set ${cwd}/PCA5_0.5_Morgan_test.bin --scaler ${cwd}/PCA5_0.5_Morgan_scaler.bin > ${name}.out 2>&1

cd ..
touch ${name}.done

EOF

    else
        echo "partial_state_model.bin found"
    	cat > ${name}.sub <<EOF
#! /bin/bash
#SBATCH -t 0-23:59:59
#SBATCH -J ${errorname}_PCA5_0.5_Morgan_${name}
#SBATCH -N 1
#SBATCH -n 80
#SBATCH --account=rrg-fekl-ac
#SBATCH --error=${name}.e%J        # The file where run time errors will be dumped
#SBATCH --output=${name}.o%J               # The file where the output of the terminal will be dumped

cd $(pwd)/$name

/scinet/niagara/software/2019b/opt/base/python/3.11.5/bin/python3 ${cwd}/main.py --save_path ${path}  --settings ${path}/${name}.json --train_set ${cwd}/PCA5_0.5_Morgan_train.bin --test_set ${cwd}/PCA5_0.5_Morgan_test.bin --scaler ${cwd}/PCA5_0.5_Morgan_scaler.bin --resume_file ${path}/partial_state_model.bin > ${name}.out 2>&1

cd ..
touch ${name}.done

EOF


 
    fi
    echo "Done ${name}.sub"
done
