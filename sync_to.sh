#!/bin/bash

# Check if destination argument is provided
if [ $# -eq 0 ]; then
  echo "Error: No destination specified"
  echo "Usage: $0 [frank|dory]"
  exit 1
fi

DESTINATION=$1

# Check if sshpass is installed
if ! command -v sshpass &>/dev/null; then
  echo "Error: sshpass is not installed"
  echo "Install it with: brew install sshpass (macOS) or apt-get install sshpass (Linux)"
  exit 1
fi

# Configuration based on destination
case $DESTINATION in
  frank)
    REMOTE_USER="frank"
    REMOTE_HOST="192.168.200.1"
    REMOTE_PATH="~/gpu_mode"
    PASSWORD="macminihardware"
    ;;
  dory)
    REMOTE_USER="dory"
    REMOTE_HOST="100.116.69.48"
    REMOTE_PATH="~/gpu_mode"
    PASSWORD="1234"
    ;;
  *)
    echo "Error: Invalid destination '$DESTINATION'"
    echo "Usage: $0 [frank|dory]"
    exit 1
    ;;
esac

# Copy current directory contents to remote
echo "Copying files to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}..."
sshpass -p "${PASSWORD}" rsync -av --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' -e ssh ./ ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/

if [ $? -eq 0 ]; then
  echo "Successfully copied files to remote server"
else
  echo "Error: Failed to copy files"
  exit 1
fi
