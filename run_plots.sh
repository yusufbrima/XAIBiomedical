#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 source_directory target_directory"
    exit 1
fi

# Assign arguments to variables
source_directory=$1
target_directory=$2

# Check if the target directory exists, and delete it if it does
if [ -d "$target_directory" ]; then
    rm -rf "$target_directory"
fi

# Create the target directory
mkdir -p "$target_directory"

# Convert .tiff files to .png and place them in the target directory
for file in "$source_directory"/*.tiff; do
    convert "$file" "$target_directory"/$(basename "${file%.tiff}").png
done
