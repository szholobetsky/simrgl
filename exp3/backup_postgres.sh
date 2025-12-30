#!/bin/bash
# Backup PostgreSQL Vector Collections for exp3
# This script backs up the vectors schema containing all experiment embeddings

echo "============================================================"
echo "Backing up PostgreSQL Vector Collections (exp3)"
echo "============================================================"

# Configuration
CONTAINER_NAME="semantic_vectors_db"
DB_NAME="semantic_vectors"
DB_USER="postgres"
SCHEMA_NAME="vectors"
BACKUP_DIR="./experiment_results/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/exp3_vectors_backup_${TIMESTAMP}.sql"

# Create backup directory if it doesn't exist
if [ ! -d "$BACKUP_DIR" ]; then
    echo "Creating backup directory: $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
fi

echo ""
echo "Backup Configuration:"
echo "  Container: $CONTAINER_NAME"
echo "  Database: $DB_NAME"
echo "  Schema: $SCHEMA_NAME"
echo "  Backup file: $BACKUP_FILE"
echo ""

# Check if container is running
if ! podman ps --filter name=$CONTAINER_NAME --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "[ERROR] Container $CONTAINER_NAME is not running!"
    echo "Please start PostgreSQL first: ./start_postgres.sh"
    exit 1
fi

echo "[1/2] Creating backup..."
podman exec $CONTAINER_NAME pg_dump -U $DB_USER -d $DB_NAME -n $SCHEMA_NAME --clean --if-exists > "$BACKUP_FILE"

if [ $? -ne 0 ]; then
    echo "[ERROR] Backup failed!"
    exit 1
fi

echo "[2/2] Verifying backup..."
BACKUP_SIZE=$(stat -f%z "$BACKUP_FILE" 2>/dev/null || stat -c%s "$BACKUP_FILE" 2>/dev/null)

if [ -z "$BACKUP_SIZE" ] || [ "$BACKUP_SIZE" -lt 1000 ]; then
    echo "[ERROR] Backup file is too small ($BACKUP_SIZE bytes). Something went wrong!"
    exit 1
fi

echo ""
echo "============================================================"
echo "[SUCCESS] Backup completed successfully!"
echo "============================================================"
echo "  File: $BACKUP_FILE"
echo "  Size: $(du -h "$BACKUP_FILE" | cut -f1)"
echo ""
echo "To restore this backup, run:"
echo "  ./restore_postgres.sh \"$BACKUP_FILE\""
echo "============================================================"

exit 0
