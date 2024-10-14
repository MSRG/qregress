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

# Function to output a file before timeout
output_before_timeout() {
    local threshold=3  # seconds (5 minutes) before timeout
    while true; do
        local time_limit=$(scontrol show job $SLURM_JOB_ID | grep -oP '(?<=TimeLimit=)\S+')
        local time_limit_sec=$(scontrol show job $SLURM_JOB_ID | grep -oP '(?<=TimeLimit=)\d+:\d+:\d+' | awk -F: '{ print ($1*3600) + ($2*60) + $3 }')
        local time_runtime=$(scontrol show job $SLURM_JOB_ID | grep -oP '(?<=RunTime=)\S+')
        local time_runtime_sec=$(scontrol show job $SLURM_JOB_ID | grep -oP '(?<=RunTime=)\d+:\d+:\d+' | awk -F: '{ print ($1*3600) + ($2*60) + $3 }')
        local timeleft=$(expr $time_limit_sec - $time_runtime_sec)
        echo $timeleft
        if [ $timeleft -lt $threshold ]; then
            echo "Job is about to timeout in less than $threshold seconds!" >> warning_output.txt
            break
        fi

        sleep 1  # Check every minute
    done
}

# Run your job commands in the background
output_before_timeout &


module load apptainer
cd $(pwd)/$name

apptainer run -C -B ${cwd} /home/gjones/projects/def-jacobsen/gjones/deb.sif /opt/miniconda/bin/python ${cwd}/main.py --save_path ${path}  --settings ${path}/${name}.json --train_set ${cwd}/5_DDCC_train.bin --test_set ${cwd}/5_DDCC_test.bin --scaler ${cwd}/5_DDCC_scaler.bin >> ${name}.out 2>&1

cd ..
touch ${name}.done

EOF
    fi
    echo "Done ${name}.sub"
done