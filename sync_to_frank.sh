#!/bin/bash

# Check if sshpass is installed
if ! command -v sshpass &> /dev/null; then
    echo "Error: sshpass is not installed"
    echo "Install it with: brew install sshpass (macOS) or apt-get install sshpass (Linux)"
    exit 1
fi

# Configuration
REMOTE_USER="frank"
REMOTE_HOST="192.168.200.1"
REMOTE_PATH="~/gpu_mode"
PASSWORD="macminihardware"

# Copy current directory contents to remote
echo "Copying files to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}..."
sshpass -p "${PASSWORD}" scp -r ./* ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/

if [ $? -eq 0 ]; then
    echo "Successfully copied files to remote server"
else
    echo "Error: Failed to copy files"
    exit 1
fi
