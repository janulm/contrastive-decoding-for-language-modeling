#!/bin/bash

# Define the base directory containing the downloaded OSF data
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../data/osf_download" && pwd)"

echo "Extracting ZIP files in: $BASE_DIR"

# Find all zip files in the directory and extract them
find "$BASE_DIR" -type f -name "*.zip" | while read -r zipfile; do
    echo "Unpacking: $zipfile"
    unzip -o "$zipfile" -d "$(dirname "$zipfile")"
done

echo "Extraction complete."
