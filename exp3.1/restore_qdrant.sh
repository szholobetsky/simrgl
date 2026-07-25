#!/bin/bash
# Restore Qdrant collections from snapshots
# Reads snapshot files from qdrant_snapshots folder

echo "================================================================================"
echo "Qdrant Restore - Loading Snapshots"
echo "================================================================================"
echo ""

# Check backup directory exists
if [ ! -d "qdrant_snapshots" ]; then
    echo "ERROR: qdrant_snapshots folder not found"
    echo "Please copy the backup folder to this directory first."
    exit 1
fi

# Check Qdrant is running
if ! curl -s http://localhost:6333/collections >/dev/null 2>&1; then
    echo "Qdrant is not running. Attempting to start..."
    podman-compose up -d
    sleep 5
fi

# Check again
if ! curl -s http://localhost:6333/collections >/dev/null 2>&1; then
    echo "ERROR: Qdrant is still not running. Please start it manually:"
    echo "  podman-compose up -d"
    exit 1
fi

echo "Restoring all collections..."
echo ""

python3 backup_restore_qdrant.py --action restore --input qdrant_snapshots
if [ $? -ne 0 ]; then
    echo "ERROR: Restore failed"
    exit 1
fi

echo ""
echo "================================================================================"
echo "SUCCESS! Restore Complete"
echo "================================================================================"
echo "Collections restored from: qdrant_snapshots/"
echo ""
echo "You can now run experiments: ./run_experiments.py"
echo "================================================================================"
