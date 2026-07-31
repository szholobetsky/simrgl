#!/bin/bash
# Launches run_cross_experiment.sh as a detached background process,
# independent of this terminal/session - see run_cross_experiment.sh's
# header for what it runs and how to monitor it once launched.
cd "$(dirname "$0")"

if pgrep -f "run_cross_experiment.sh" > /dev/null; then
  echo "Already running:"
  pgrep -af "run_cross_experiment.sh"
  echo "Not starting a second copy."
  exit 1
fi

nohup ./run_cross_experiment.sh > /dev/null 2>&1 &
disown
echo "Launched. Monitor with:"
echo "  cat experiment_results/RUNNING.txt"
echo "  tail -f logs/cross_run_*.log"
echo "  cat experiment_results/STATUS.md"
echo "  ./monitor.sh              # one-shot status dump"
echo "  ./monitor_dashboard.sh    # continuous three-level progress bar"
