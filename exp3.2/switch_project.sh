#!/bin/bash
#
# Switch the shared Postgres vector store from one project to the next.
#
# What it does, in order:
#   1. Verify the previous project's checkpoint shows nothing in-flight
#      (warns and asks for confirmation if not - never silently discards
#      an unfinished run).
#   2. Optionally back up the current vectors schema with pg_dump - but
#      only if it's under a size threshold (disk isn't infinite; the
#      vectors are 100% reproducible from the source SQLite DB via ETL,
#      so a backup here is a convenience, not a safety net).
#   3. Drop and recreate the 'vectors' schema (non-interactive - unlike
#      clear_postgres_vectors.py, this is meant to run unattended).
#   4. Verify the schema is actually empty afterwards.
#
# Results (CSV/JSON/checkpoint/logs under experiment_results/<project>/)
# are plain files, never touched by this script - Postgres only ever
# holds reproducible intermediate vectors.
#
# Usage:
#   ./switch_project.sh --from sonar --to celery [--task-unit ticket] \
#       [--backup] [--backup-max-mb 2000] [--yes]

set -euo pipefail

CONTAINER_NAME="semantic_vectors_db"
DB_NAME="semantic_vectors"
DB_USER="postgres"
SCHEMA_NAME="vectors"
TASK_UNIT="ticket"
DO_BACKUP=false
BACKUP_MAX_MB=2000
ASSUME_YES=false
FROM_PROJECT=""
TO_PROJECT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM_PROJECT="$2"; shift 2 ;;
        --to) TO_PROJECT="$2"; shift 2 ;;
        --task-unit) TASK_UNIT="$2"; shift 2 ;;
        --backup) DO_BACKUP=true; shift ;;
        --backup-max-mb) BACKUP_MAX_MB="$2"; shift 2 ;;
        --yes) ASSUME_YES=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$TO_PROJECT" ]; then
    echo "Usage: $0 --from <project|none> --to <project> [--task-unit ticket] [--backup] [--yes]"
    exit 1
fi

echo "================================================================"
echo "Project switch: ${FROM_PROJECT:-<none>} -> ${TO_PROJECT} (task_unit=${TASK_UNIT})"
echo "================================================================"

# --- 1. Verify previous project's checkpoint is complete -------------------
if [ -n "$FROM_PROJECT" ] && [ "$FROM_PROJECT" != "none" ]; then
    CHECKPOINT="experiment_results/${FROM_PROJECT}/${TASK_UNIT}/checkpoint.json"
    if [ -f "$CHECKPOINT" ]; then
        FAILED_ETL=$(python3 -c "import json; d=json.load(open('$CHECKPOINT')); print(len(d.get('failed_etl', [])))")
        FAILED_EXP=$(python3 -c "import json; d=json.load(open('$CHECKPOINT')); print(len(d.get('failed_experiments', [])))")
        DONE_ETL=$(python3 -c "import json; d=json.load(open('$CHECKPOINT')); print(len(d.get('completed_etl', [])))")
        DONE_EXP=$(python3 -c "import json; d=json.load(open('$CHECKPOINT')); print(len(d.get('completed_experiments', [])))")
        echo "[1/4] ${FROM_PROJECT} checkpoint: ${DONE_EXP} experiments done, ${FAILED_ETL} ETL failures, ${FAILED_EXP} experiment failures"

        if [ "$DONE_ETL" != "$DONE_EXP" ] || [ "$FAILED_ETL" != "0" ] || [ "$FAILED_EXP" != "0" ]; then
            echo "  WARNING: ${FROM_PROJECT} looks incomplete or has failures."
            if [ "$ASSUME_YES" = false ]; then
                read -p "  Clear Postgres and move to ${TO_PROJECT} anyway? (yes/no): " -r
                [[ $REPLY == "yes" ]] || { echo "Aborted."; exit 1; }
            fi
        fi
    else
        echo "[1/4] No checkpoint found at $CHECKPOINT - nothing to verify, continuing."
    fi
else
    echo "[1/4] No previous project given - skipping checkpoint check."
fi

# --- 2. Optional size-gated backup ------------------------------------------
if podman ps --filter name=$CONTAINER_NAME --format "{{.Names}}" 2>/dev/null | grep -q "^${CONTAINER_NAME}$"; then
    DB_SIZE_MB=$(podman exec $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME -t -c \
        "SELECT pg_database_size('${DB_NAME}') / 1024 / 1024;" 2>/dev/null | tr -d '[:space:]')
    echo "[2/4] Current Postgres DB size: ${DB_SIZE_MB:-unknown}MB"

    if [ "$DO_BACKUP" = true ]; then
        if [ -n "$DB_SIZE_MB" ] && [ "$DB_SIZE_MB" -gt "$BACKUP_MAX_MB" ]; then
            echo "  Skipping backup: ${DB_SIZE_MB}MB > --backup-max-mb ${BACKUP_MAX_MB}MB."
            echo "  (Vectors are reproducible from the source .db via ETL - re-run instead of restoring.)"
        else
            mkdir -p "experiment_results/${FROM_PROJECT}/${TASK_UNIT}"
            BACKUP_FILE="experiment_results/${FROM_PROJECT}/${TASK_UNIT}/postgres_backup_$(date +%Y%m%d_%H%M%S).sql"
            echo "  Backing up to $BACKUP_FILE ..."
            podman exec $CONTAINER_NAME pg_dump -U $DB_USER -d $DB_NAME -n $SCHEMA_NAME --clean --if-exists > "$BACKUP_FILE"
            echo "  Backup done: $(du -h "$BACKUP_FILE" | cut -f1)"
        fi
    else
        echo "  --backup not set, skipping (results CSV/JSON already persisted separately)."
    fi

    # --- 3. Clear the schema (non-interactive) ------------------------------
    echo "[3/4] Dropping and recreating schema '${SCHEMA_NAME}'..."
    podman exec $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME -c \
        "DROP SCHEMA IF EXISTS ${SCHEMA_NAME} CASCADE; CREATE SCHEMA ${SCHEMA_NAME};" > /dev/null

    # --- 4. Verify empty -----------------------------------------------------
    REMAINING=$(podman exec $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME -t -c \
        "SELECT count(*) FROM information_schema.tables WHERE table_schema='${SCHEMA_NAME}';" | tr -d '[:space:]')
    if [ "$REMAINING" != "0" ]; then
        echo "  ERROR: schema still has $REMAINING table(s) after clearing!"
        exit 1
    fi
    echo "[4/4] Verified: '${SCHEMA_NAME}' schema is empty."
else
    echo "[2-4/4] Postgres container '${CONTAINER_NAME}' is not running - nothing to clear."
fi

echo ""
echo "================================================================"
echo "Ready for ${TO_PROJECT}. Next:"
echo "  python etl_pipeline.py --project ${TO_PROJECT} ..."
echo "================================================================"
