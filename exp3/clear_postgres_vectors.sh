#!/bin/bash

echo "============================================================"
echo "CLEAR POSTGRESQL VECTOR COLLECTIONS"
echo "============================================================"
echo ""
echo "Database: localhost:5432/semantic_vectors"
echo "Schema: vectors"
echo ""
echo "WARNING: This will delete ALL vector collections and rawdata!"
echo "============================================================"
echo ""

python3 clear_postgres_vectors.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to clear PostgreSQL vectors"
    exit 1
fi

echo ""
echo "============================================================"
echo "SUCCESS: All vector collections cleared"
echo "============================================================"
echo ""
echo "You can now run:"
echo "  ./run_etl_dual_postgres.sh"
echo ""
echo "to create fresh DUAL collections (RECENT + ALL)"
echo "============================================================"
