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
#SBATCH -t 0-23:59:59
#SBATCH -J ${errorname}_sine_${name}
#SBATCH -N 1
#SBATCH -n 80
#SBATCH --account=rrg-fekl-ac
#SBATCH --error=${name}.e%J        # The file where run time errors will be dumped
#SBATCH --output=${name}.o%J               # The file where the output of the terminal will be dumped

module load apptainer
cd $(pwd)/$name

apptainer run -C -B ${cwd} /home/j/jacobsen/gjones/deb.sif /opt/miniconda/bin/python ${cwd}/main.py --save_path ${path}  --settings ${path}/${name}.json --train_set ${cwd}/sine_train.bin --test_set ${cwd}/sine_test.bin --scaler ${cwd}/sine_scaler.bin >> ${name}.out 2>&1

cd ..
touch ${name}.done

EOF
    fi
    echo "Done ${name}.sub"
done
