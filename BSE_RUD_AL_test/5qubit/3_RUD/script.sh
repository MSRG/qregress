#!/bin/bash

# Loop through all files in the current directory
for file in *; do
    # Check if the item is a regular file (not a directory)
    if [ -f "$file" ]; then
        # Get the file name without extension
        file_name=$(basename -- "$file")
        folder_name="${file_name%.*}"

        # Create the folder and move the file to it
        mkdir "$folder_name" 2>/dev/null # Redirect error output to /dev/null to suppress errors if the folder already exists
        mv "$file" "$folder_name/$file"

        echo "File moved to: $folder_name/$file"
    fi
done

