#! /bin/bash

cwd=$(pwd)
errorname=$(basename "$cwd")
echo "$cwd"

for dir in 0.1 0.3 0.5 0.7 0.8; do
    name="A2_HWE-CNOT"
    path="${cwd}/${dir}/${name}"
    json_file="${path}/${name}.json"
    resume_file=""

    # Determine the resume file if it exists
    if [ -f "${path}/final_state_model.bin" ]; then
        resume_file="--resume_file ${path}/final_state_model.bin"
    elif [ -f "${path}/partial_state_model.bin" ]; then
        resume_file="--resume_file ${path}/partial_state_model.bin"
    else
        resume_file=""
    fi

    # Determine the wall time based on MAX_ITER
    if [ -f "$json_file" ]; then
        max_iter=$(jq '.MAX_ITER' "$json_file")  # Extract MAX_ITER using jq
        if (( max_iter < 10 )); then
            wall_time="4:00:00"   # 4 hours
        else
            wall_time="4-00:00:00"  # 2 days
        fi
    else
        wall_time="4-00:00:00"
    fi

    # Generate the SLURM submission script
    cat > "${path}/${name}.sub" <<EOF
#! /bin/bash
#SBATCH -t $wall_time
#SBATCH -J ${dir}_${name}
#SBATCH -N 1
#SBATCH --cpus-per-task 64
#SBATCH --account=rrg-jacobsen-ab
#SBATCH --error=${name}.e%J
#SBATCH --output=${name}.o%J

export OMP_NUM_THREADS=64

cd $path

/lustre06/project/6006115/gjones/env/bin/python3 ${path}/main.py \
    --save_path $path \
    --settings $json_file \
    --train_set ${cwd}/${dir}/${dir}_5_DDCC_train.bin \
    --test_set ${cwd}/${dir}/${dir}_5_DDCC_test.bin \
    --scaler ${cwd}/${dir}/${dir}_5_DDCC_scaler.bin \
    --save_circuits True \
    $resume_file > ${name}.out 2>&1

cd ..
touch ${name}.done
EOF

    echo "Done ${name}.sub"
done

