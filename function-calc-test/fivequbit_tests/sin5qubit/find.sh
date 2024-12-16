#!/bin/bash
cwd=$(pwd)
errorname=$(basename `pwd`)
echo $cwd

# Define the directory to search (current directory by default)
SEARCH_DIR="."

# Find all model_log.csv files created in December
DECEMBER_LOGS=$(find "$SEARCH_DIR" -type f -name "model_log.csv" -newermt "2024-12-01" ! -newermt "2025-01-01")

# Loop through all `model_log.csv` files in the directory
for FILE in $(find "$SEARCH_DIR" -type f -name "model_log.csv"); do
    # Check if the file is in the DECEMBER_LOGS list
    if echo "$DECEMBER_LOGS" | grep -q "$FILE"; then
        echo "Skipping December file: $FILE"
        continue
    fi

    echo "Processing file: $FILE"
    # Add your processing commands here
    name=$(dirname "${FILE#./}")
    settings_folder=${name#M-A1-CNOT_Efficient-CRX_}
    path="${cwd}/${name}"
    cd $(pwd)/$name
    echo "${name}" 
    python ${cwd}/main.py --save_path ${path}  --settings ${path}/${name}.json --train_set ${cwd}/sine_train.bin --test_set ${cwd}/sine_test.bin --scaler ${cwd}/sine_scaler.bin --save_circuits True 
    
    cd ..
    touch ${name}.done

done

