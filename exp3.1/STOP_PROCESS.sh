#!/bin/bash
# Controlled stop for the exp3.1 multi-project sweep.
#
# Safe to run at any time - checkpoint.json is only updated after a variant
# fully completes (ETL upsert to Postgres + experiment eval), so killing
# mid-run only loses the currently in-flight variant's GPU work, never
# corrupts already-completed rows. Re-launch later with ./run.sh - the
# flock in run_full_experiment.sh releases automatically on process exit,
# so a clean re-launch is never blocked by this stop.

set -u

PIDS=$(pgrep -f "run_full_experiment.sh|run_comprehensive_experiments.py")

if [ -z "$PIDS" ]; then
  echo "Nothing running - already stopped."
  exit 0
fi

echo "Stopping:"
pgrep -af "run_full_experiment.sh|run_comprehensive_experiments.py"

kill $PIDS
sleep 2

STILL=$(pgrep -f "run_full_experiment.sh|run_comprehensive_experiments.py")
if [ -n "$STILL" ]; then
  echo "Still alive after SIGTERM, forcing:"
  pgrep -af "run_full_experiment.sh|run_comprehensive_experiments.py"
  kill -9 $STILL
  sleep 1
fi

if pgrep -f "run_full_experiment.sh|run_comprehensive_experiments.py" > /dev/null; then
  echo "WARNING: something is still running - check manually with:"
  echo "  pgrep -af run_full_experiment.sh"
  echo "  pgrep -af run_comprehensive_experiments.py"
  exit 1
fi

echo "Stopped. Progress is safe in checkpoint.json - resume anytime with ./run.sh"
