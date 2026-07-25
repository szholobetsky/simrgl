#!/bin/bash
# Continuous three-level progress dashboard (Ctrl-C to stop).
# Usage: ./monitor_dashboard.sh [--interval N]   (default: 10s)
cd "$(dirname "$0")"
python3.12 monitor_dashboard.py --watch "$@"
