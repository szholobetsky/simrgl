#!/bin/bash
#
# Ensures the Postgres container (semantic_vectors_db) is up and accepting
# connections. Run this after a reboot, before ./run.sh - a full machine
# restart does NOT bring podman containers back automatically, even with
# `restart: unless-stopped` in postgres-compose.yml (that policy only
# applies while podman itself keeps running, not across a reboot).
#
# Safe to run any time, including while nothing is broken - it's a no-op
# if the container is already up.

set -uo pipefail

CONTAINER_NAME="semantic_vectors_db"

cd "$(dirname "$0")"

STATUS=$(podman ps -a --filter name="^${CONTAINER_NAME}$" --format "{{.Status}}" 2>/dev/null)

if [ -z "$STATUS" ]; then
  echo "Container '${CONTAINER_NAME}' doesn't exist yet - creating via postgres-compose.yml..."
  podman-compose -f postgres-compose.yml up -d
elif echo "$STATUS" | grep -qi "^up"; then
  echo "Container '${CONTAINER_NAME}' already running: ${STATUS}"
else
  echo "Container '${CONTAINER_NAME}' exists but is stopped (${STATUS}) - starting..."
  podman start "${CONTAINER_NAME}"
fi

echo "Waiting for Postgres to accept connections..."
for i in $(seq 1 30); do
  if podman exec "${CONTAINER_NAME}" pg_isready -U postgres > /dev/null 2>&1; then
    echo "Postgres is ready."
    exit 0
  fi
  sleep 1
done

echo "ERROR: Postgres did not become ready within 30s - check 'podman logs ${CONTAINER_NAME}'"
exit 1
