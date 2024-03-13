#! /bin/bash

for i in */; do
    name=${i%/}
    
    # Extracting the parent directory name
    settings_folder=${name#M-A1-CNOT_Efficient-CRX_}
    
    cat > ${name}.sub <<EOF

cd $(pwd)/$name
python ~/qregress/main.py --settings ${name}.json --train_set ~/qregress/database/processed/16feats/${settings_folder}/BSE49_16feats_train.bin --test_set ~/qregress/database/processed/16feats/${settings_folder}/BSE49_16feats_test.bin --scaler ~/qregress/database/processed/16feats/${settings_folder}/BSE49_16feats_scaler.bin >> ${name}.out 2>&1

cd ..
touch ${name}.done

EOF
    echo "Done ${name}.sub"
done
