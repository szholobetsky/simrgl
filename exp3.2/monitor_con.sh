#!/bin/bash
# Continuous monitoring: re-runs monitor.sh every 30s.
cd "$(dirname "$0")"
watch -n 30 ./monitor.sh
