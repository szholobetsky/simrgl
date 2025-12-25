#!/bin/bash
# Restore PostgreSQL Vector Collections
# This script restores the vectors schema from a backup file

echo "============================================================"
echo "Restoring PostgreSQL Vector Collections"
echo "============================================================"

# Configuration
CONTAINER_NAME="semantic_vectors_db"
DB_NAME="semantic_vectors"
DB_USER="postgres"
SCHEMA_NAME="vectors"

# Get backup file from command line argument or use latest
if [ -z "$1" ]; then
    echo "No backup file specified. Looking for latest backup..."
    BACKUP_FILE=$(ls -t ./backups/vectors_backup_*.sql 2>/dev/null | head -n 1)
    if [ -z "$BACKUP_FILE" ]; then
        echo "[ERROR] No backup files found in ./backups/"
        echo "Usage: $0 [backup_file.sql]"
        exit 1
    fi
    echo "Found latest backup: $BACKUP_FILE"
else
    BACKUP_FILE="$1"
    if [ ! -f "$BACKUP_FILE" ]; then
        echo "[ERROR] Backup file not found: $BACKUP_FILE"
        exit 1
    fi
fi

echo ""
echo "Restore Configuration:"
echo "  Container: $CONTAINER_NAME"
echo "  Database: $DB_NAME"
echo "  Schema: $SCHEMA_NAME"
echo "  Backup file: $BACKUP_FILE"
echo ""

# Check if container is running
if ! podman ps --filter name=$CONTAINER_NAME --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "[ERROR] Container $CONTAINER_NAME is not running!"
    echo "Please start PostgreSQL first: podman start $CONTAINER_NAME"
    exit 1
fi

echo "[WARNING] This will DROP and recreate the '$SCHEMA_NAME' schema!"
read -p "Press Enter to continue or Ctrl+C to cancel... "

echo ""
echo "[1/2] Restoring from backup..."
cat "$BACKUP_FILE" | podman exec -i $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME

if [ $? -ne 0 ]; then
    echo "[ERROR] Restore failed!"
    exit 1
fi

echo "[2/2] Verifying restore..."
podman exec $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME -c "\dt ${SCHEMA_NAME}.*"

echo ""
echo "============================================================"
echo "[SUCCESS] Restore completed successfully!"
echo "============================================================"
echo "  Restored from: $BACKUP_FILE"
echo "  Schema: $SCHEMA_NAME"
echo ""
echo "You can now use the restored collections in your RAG system."
echo "============================================================"

exit 0
