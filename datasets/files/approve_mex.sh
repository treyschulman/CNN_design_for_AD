#!/bin/bash

# Set this to your SPM folder
SPM_DIR="/Users/treyschulman/cs598dlh/project/spm"

echo "Finding all .mexmaci64 files inside $SPM_DIR..."
find "$SPM_DIR" -name "*.mexmaci64" | while read file; do
    echo "Approving $file..."
    xattr -d com.apple.quarantine "$file"
done

echo "✅ All .mexmaci64 files unquarantined."
