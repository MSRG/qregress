#! /bin/bash

for i in */; do
    name=${i%/}
    
    # Extracting the parent directory name
    settings_folder=${name#M-A1-CNOT_Efficient-CRX_}
    
    cat > ${name}.sub <<EOF
#! /bin/bash
#SBATCH -t 23:59:00
#SBATCH -J ${name}
#SBATCH -N 1
#SBATCH -n 80
#SBATCH --account=def-fekl

module load python/3.9.8
source /home/j/jacobsen/vikikrpd/Softwares/qregress/qregress_env/bin/activate 

cd $(pwd)/$name

python /home/j/jacobsen/vikikrpd/Softwares/qregress/main.py --settings ${name}.json --train_set /home/j/jacobsen/vikikrpd/Softwares/qregress/database/processed/16feats/${settings_folder}/BSE49_16feats_train.bin --test_set /home/j/jacobsen/vikikrpd/Softwares/qregress/database/processed/16feats/${settings_folder}/BSE49_16feats_test.bin --scaler /home/j/jacobsen/vikikrpd/Softwares/qregress/database/processed/16feats/${settings_folder}/BSE49_16feats_scaler.bin >> ${name}.out 2>&1

cd ..
touch ${name}.done

EOF
    echo "Done ${name}.sub"
done
