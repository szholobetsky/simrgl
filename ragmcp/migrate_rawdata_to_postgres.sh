#!/bin/bash

echo "============================================================"
echo "MIGRATE RAWDATA: SQLite to PostgreSQL"
echo "============================================================"
echo ""
echo "This script migrates RAWDATA table from SQLite to PostgreSQL"
echo ""
echo "Source: ../data/sonar.db (SQLite)"
echo "Target: PostgreSQL (localhost:5432/semantic_vectors)"
echo "Schema: vectors.rawdata"
echo ""
echo "Fields migrated:"
echo "  - TASK_NAME"
echo "  - PATH"
echo "  - MESSAGE"
echo "  - DIFF"
echo ""
echo "Estimated time: 2-5 minutes (depends on data size)"
echo "============================================================"
echo ""

python3 migrate_rawdata_to_postgres.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Migration failed"
    exit 1
fi

echo ""
echo "============================================================"
echo "SUCCESS: RAWDATA migration completed"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Verify data in PostgreSQL:"
echo "   psql -h localhost -U postgres -d semantic_vectors"
echo "   SELECT COUNT(*) FROM vectors.rawdata;"
echo ""
echo "2. Run the two-phase agent:"
echo "   ./two_phase_agent.sh"
echo "============================================================"
