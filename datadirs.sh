#!/bin/bash
# BSE-calc-test
# BSE5_real
# BSE_RUD_AL_test
# BSE_learning_curves
# function-calc-test
# qml_DDCC

for dir in BSE-calc-test BSE5_real BSE_RUD_AL_test BSE_learning_curves function-calc-test qml_DDCC; do
    for file in ./helperfiles/*; do
        filename=$(basename "$file")  # Extract just the filename
        find "$dir" -name "$filename" -exec rm -rf {} \;
    done
    find "$dir" -name "helperfiles" -exec rm -rf {} \;
done

