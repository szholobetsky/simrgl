#!/bin/bash
# One-shot status dump for the background run_full_experiment.sh run.
# For continuous monitoring, wrap it: watch -n 30 ./monitor.sh
cd "$(dirname "$0")"

echo "================================================================"
echo "Process"
echo "================================================================"
if pgrep -f "run_full_experiment.sh" > /dev/null; then
  ps -o pid,etime,cmd -p "$(pgrep -f run_full_experiment.sh | head -1)"
  echo ""
  echo "Active step process:"
  pgrep -af "run_comprehensive_experiments.py" || echo "  (between steps - switching project / writing status)"
else
  echo "NOT RUNNING (no run_full_experiment.sh process found)"
fi

echo ""
echo "================================================================"
echo "Current step"
echo "================================================================"
if [ -f experiment_results/RUNNING.txt ]; then
  cat experiment_results/RUNNING.txt
else
  echo "(no experiment_results/RUNNING.txt yet)"
fi

echo ""
echo "================================================================"
echo "Per-(project,task_unit) status"
echo "================================================================"
if [ -f experiment_results/STATUS.md ]; then
  cat experiment_results/STATUS.md
else
  echo "(no experiment_results/STATUS.md yet)"
fi

echo ""
echo "================================================================"
echo "Latest log tail (logs/full_run_*.log)"
echo "================================================================"
LATEST_LOG=$(ls -t logs/full_run_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
  echo "File: $LATEST_LOG"
  tail -n 15 "$LATEST_LOG" | grep -vE "^\s*(Batches|Upserting|Evaluating|Encoding batches):"
else
  echo "(no log file yet)"
fi

echo ""
echo "================================================================"
echo "GPU"
echo "================================================================"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "(nvidia-smi unavailable)"

echo ""
echo "================================================================"
echo "Postgres"
echo "================================================================"
podman ps --filter name=semantic_vectors_db --format "{{.Names}}: {{.Status}}" 2>/dev/null || echo "(podman unavailable)"
