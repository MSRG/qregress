#!/bin/bash
# Modify when it's time to clean up the directory
for dir in $(find . -type d -mindepth 1 -maxdepth 1 ! -name '.*' ! -path './helperfiles' ! -path './__pycache__'); do
    echo "Processing: $dir"
    # Add your commands here
    for file in ./helperfiles/*; do
        filename=$(basename "$file")  # Extract just the filename
        echo "$(find "$dir" -name "$filename")"
    done
done

