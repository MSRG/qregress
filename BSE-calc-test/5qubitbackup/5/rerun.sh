#!/bin/bash
# Grab iteration information
dir=$(find . -maxdepth 2 -name "model*.csv")

# Loop over directories
for i in $dir; do
    path=$(dirname "$i")
    echo "$path"
    sumiter=0

    # Loop over existing model_log.csv files to find the final iterations
    for j in "${path}"/*model_log.csv; do
        echo "$j"
        raniter=$(awk -F',' 'END {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2}' "$j")

        # Ensure raniter is a valid integer
        if [[ "$raniter" =~ ^[0-9]+$ ]]; then
            sumiter=$((sumiter + raniter + 1))
            echo "$j has $raniter iterations"
            echo "$sumiter"
        else
            echo "Warning: Could not extract valid iteration from $j"
        fi
    done

    # Write the remaining iterations to file
    remain=$((1000 - sumiter))
    echo "Iterations left: $remain"

    # Uncomment these lines to apply changes if needed:
    # mv "${path}/model_log.csv" "${path}/1_model_log.csv"
    # sed -i 's/"MAX_ITER": 1000/"MAX_ITER": '$remain'/g' "${path}"/*.json

    echo "$(cat "${path}"/*.json)"
    echo
done

