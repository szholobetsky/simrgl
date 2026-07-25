#!/bin/bash
# Launches run_full_experiment.sh as a detached background process,
# independent of this terminal/session - see run_full_experiment.sh's
# header for what it runs and how to monitor it once launched.
cd "$(dirname "$0")"
nohup ./run_full_experiment.sh > /dev/null 2>&1 &
disown
echo "Launched. Monitor with:"
echo "  cat experiment_results/RUNNING.txt"
echo "  tail -f logs/full_run_*.log"
echo "  cat experiment_results/STATUS.md"
