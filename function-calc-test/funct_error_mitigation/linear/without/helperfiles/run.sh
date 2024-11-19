#! /bin/bash
cwd=$(pwd)
errorname=$(basename `pwd`)
echo $cwd
for i in */; do
    name=${i%/}
    # Extracting the parent directory name
    settings_folder=${name#M-A1-CNOT_Efficient-CRX_}
    path="${cwd}/${name}"
    if [ ! -f ${path}/${name}_results.json ]; then
        echo "${name}.done not found!"
    	cat > ${name}.sub <<EOF
#! /bin/bash
#SBATCH -t 7-00:00:00
#SBATCH -J ${errorname}_DDCC_${name}
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem-per-cpu=300           # memory per cpu
#SBATCH --account=rrg-jacobsen-ab
#SBATCH --error=${name}.e%J        # The file where run time errors will be dumped
#SBATCH --output=${name}.o%J               # The file where the output of the terminal will be dumped

module load apptainer
cd $(pwd)/$name

apptainer run -C -B ${cwd} /home/gjones/projects/def-jacobsen/gjones/deb.sif /opt/miniconda/bin/python ${cwd}/main.py --save_path ${path}  --settings ${path}/${name}.json --train_set ${cwd}/5_DDCC_train.bin --test_set ${cwd}/5_DDCC_test.bin --scaler ${cwd}/5_DDCC_scaler.bin >> ${name}.out 2>&1

cd ..
touch ${name}.done

EOF
    fi
    echo "Done ${name}.sub"
done
