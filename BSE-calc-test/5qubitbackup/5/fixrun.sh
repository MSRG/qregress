#! /bin/bash
cwd=$(pwd)
errorname=$(basename "$cwd")
echo "Current working directory: $cwd"

# List of specific directories to loop over
dirs=(
    "./A1_Efficient-CRX/"
    "./A1_Efficient-CRZ/"
    "./A1_Full-CRZ/"
    "./A2-A2-CNOT_Efficient-CRZ/"
    "./A2-A2-CZ_Efficient-CRZ/"
    "./A2_ESU2/"
    "./IQP_Efficient-CRZ/"
    "./M-A1-CZ_ESU2/"
    "./M-A2-CNOT_Efficient-CRX/"
    "./M-A2-CNOT_Efficient-CRZ/"
    "./M-A2-CNOT_ESU2/"
    "./M-A2-CZ_Efficient-CRX/"
    "./M-M-CZ_Efficient-CRX/"
)

for i in "${dirs[@]}"; do
    # Remove trailing slash and extract name
    name=$(basename "$i")
    path="${cwd}/${name}"
    
    cat > "${name}.sub" <<EOF
#! /bin/bash
#SBATCH -t 2-00:00:00
#SBATCH -J ${errorname}_PCA5_0.8_Morgan_${name}
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --account=rrg-jacobsen-ab
#SBATCH --error=${name}.e%J        # The file where run time errors will be dumped
#SBATCH --output=${name}.o%J       # The file where the output of the terminal will be dumped

cd $path

/lustre06/project/6006115/gjones/env/bin/python3 ${cwd}/main.py --save_path ${path} --settings ${path}/${name}.json --train_set ${cwd}/PCA5_0.8_Morgan_train.bin --test_set ${cwd}/PCA5_0.8_Morgan_test.bin --scaler ${cwd}/PCA5_0.8_Morgan_scaler.bin --save_circuits True > ${name}.out 2>&1

cd ..
touch ${name}.done
EOF

    echo "Done ${name}.sub"
    sbatch "${name}.sub"
done

