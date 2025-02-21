#!/bin/bash
# Modify when it's time to clean up the directory
for dir in $(find . -type d -mindepth 1 -maxdepth 1 ! -name '.*' ! -path './helperfiles' ! -path './__pycache__'); do
    echo "Processing: $dir"
    # Add your commands here
    # for file in ./helperfiles/*; do
    #     filename=$(basename "$file")  # Extract just the filename
    #     echo "$(find "$dir" -name "$filename")"
    # done
    echo "$(find "$dir" -name "Quantum.py")"
    find "$dir" -name "Quantum.py" -exec cp ./helperfiles/quantum/Quantum.py {} \;
    echo "$(find "$dir" -name "main.py")"
    find "$dir" -name "main.py" -exec cp ./helperfiles/main.py {} \;
    echo "$(find "$dir" -name "Ansatz.py")"
    find "$dir" -name "Ansatz.py" -exec cp ./helperfiles/quantum/circuits/Ansatz.py {} \;
    echo "$(find "$dir" -name "Encoders.py")"
    find "$dir" -name "Encoders.py" -exec cp ./helperfiles/quantum/circuits/Encoders.py {} \;
done

