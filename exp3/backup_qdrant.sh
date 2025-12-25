#!/bin/bash
# Backup Qdrant collections to snapshots
# Creates snapshot files in qdrant_snapshots folder

echo "================================================================================"
echo "Qdrant Backup - Creating Snapshots"
echo "================================================================================"
echo ""

# Create backup directory
mkdir -p qdrant_snapshots

# Check Qdrant is running
if ! curl -s http://localhost:6333/collections >/dev/null 2>&1; then
    echo "ERROR: Qdrant is not running. Please start it first."
    echo "Run: podman-compose up -d"
    exit 1
fi

echo "Backing up all collections..."
echo ""

python3 backup_restore_qdrant.py --action backup --output qdrant_snapshots
if [ $? -ne 0 ]; then
    echo "ERROR: Backup failed"
    exit 1
fi

echo ""
echo "================================================================================"
echo "SUCCESS! Backup Complete"
echo "================================================================================"
echo "Snapshots saved to: qdrant_snapshots/"
echo ""
echo "To restore on another machine:"
echo "  1. Copy qdrant_snapshots folder to the new machine"
echo "  2. Start Qdrant: podman-compose up -d"
echo "  3. Run: ./restore_qdrant.sh"
echo "================================================================================"
