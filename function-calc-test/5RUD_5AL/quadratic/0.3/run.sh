#! /bin/bash
for i in */; do
    name=${i%/}
    # Extracting the parent directory name
    settings_folder=${name#M-A1-CNOT_Efficient-CRX_}
    path="/home/gjones/scratch/5RUD_5AL/quadratic/0.3/${name}"
    if [ ! -f ${path}/${name}_results.json ]; then
        echo "${name}.done not found!"
    	cat > ${name}.sub <<EOF
#! /bin/bash
#SBATCH -t 5-00:00:00
#SBATCH -J quadratic_0.3_${name}
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem-per-cpu=300           # memory per cpu
#SBATCH --account=rrg-jacobsen-ab
#SBATCH --error=${name}.e%J        # The file where run time errors will be dumped
#SBATCH --output=${name}.o%J               # The file where the output of the terminal will be dumped

module load apptainer 
cd $(pwd)/$name

apptainer run -C -B /home/gjones/scratch/5RUD_5AL/quadratic/0.3 /home/gjones/projects/def-jacobsen/gjones/deb.sif /opt/miniconda/bin/python /home/gjones/scratch/5RUD_5AL/quadratic/0.3/main.py --save_path ${path}  --settings ${path}/${name}.json --train_set /home/gjones/scratch/5RUD_5AL/quadratic/0.3/quadratic_train.bin --test_set /home/gjones/scratch/5RUD_5AL/quadratic/0.3/quadratic_test.bin --scaler /home/gjones/scratch/5RUD_5AL/quadratic/0.3/quadratic_scaler.bin >> ${name}.out 2>&1 

cd ..
touch ${name}.done

EOF
    fi
    echo "Done ${name}.sub"
done
