#!/bin/bash

# Define the pattern to match
PATTERN="CarlaUE4"

# Find and kill processes matching the pattern
ps aux | grep "$PATTERN" | grep -v grep | awk '{print $2}' | xargs -r kill

# Check the exit status of the kill command
if [[ $? -eq 0 ]]; then
    echo "Successfully killed all processes containing '$PATTERN'."
else
    echo "Failed to kill processes containing '$PATTERN' or no such processes found."
fi

