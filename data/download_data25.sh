#!/bin/bash

# Define the OSF project ID
PROJECT_ID="ryjfm"

# Determine the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the target directory for downloads relative to the script's location
TARGET_DIR="$SCRIPT_DIR/../../data/osf_download25"
TARGET_DIR="/cluster/home/janulm/thesis/data/osf_download25"

# Print the target directory for debugging purposes
echo "Files will be downloaded to: $TARGET_DIR"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Navigate to the target directory
cd "$TARGET_DIR"

# Clone the entire OSF project storage
osf -p "$PROJECT_ID" clone