#!/bin/bash
cwd=$(pwd)
errorname=$(basename `pwd`)
echo $cwd

for i in ./*; do
 if [ -d $i ]; then
  name=${i#./}
  settings_folder=${name#M-A1-CNOT_Efficient-CRX_}
  path="${cwd}/${name}"
  if [ ! -f "${name}/1_model_log.csv" ] && [ -f "${name}/${name}.json" ]; then
   echo "${name}"
   cat > ${name}.sub <<EOF
#! /bin/bash
#SBATCH -t 2-00:00:00
#SBATCH -J ${errorname}_PCA5_0.8_Morgan_${name}
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --account=rrg-jacobsen-ab
#SBATCH --error=${name}.e%J        # The file where run time errors will be dumped
#SBATCH --output=${name}.o%J               # The file where the output of the terminal will be dumped

cd $(pwd)/$name

/lustre06/project/6006115/gjones/env/bin/python3 ${cwd}/main.py --save_path ${path}  --settings ${path}/${name}.json --train_set ${cwd}/PCA5_0.8_Morgan_train.bin --test_set ${cwd}/PCA5_0.8_Morgan_test.bin --scaler ${cwd}/PCA5_0.8_Morgan_scaler.bin --save_circuits True  > ${name}.out 2>&1

cd ..
touch ${name}.done

EOF
   echo "Done ${name}.sub"
   sbatch "${name}.sub"
  fi
 fi
done
