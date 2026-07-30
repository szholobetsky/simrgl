#!/bin/bash
#
# exp3.2 cross-vocabulary experiment: index built from commit messages
# (title/desc/diff), queried with the held-out Jira ticket test set
# (title/desc/comments) - full 3x3 Cartesian product, not matched pairs.
# See README.md for the design rationale and EXPERIMENT_PLAN.md for the
# execution mechanics this script implements.
#
#   9 projects (no agilebill - it has zero TASK rows, so no query side at
#   all) x 2 split strategies (recent, modn) x 2 embedding models
#   (bge-small, bge-large - the only two that fit this GPU's 6GB VRAM) x
#   3 train_sources x 3 query_sources x 2 targets x 3 windows.
#
# Unlike exp3.1's run_full_experiment.sh, there is no ticket/commit
# alternation per project - task_unit=cross always touches both the TASK
# table (query side) and the COMMITS/RAWDATA table (index side) in a
# single pass per project.
#
# Meant to run for a long time - launch it detached in its own session,
# not as a subprocess of a chat session:
#   nohup ./run_cross_experiment.sh > /dev/null 2>&1 &
#   disown
# or inside tmux/screen.
#
# Fully resumable from a cold start at any point: just run this exact same
# command again. Every step is idempotent - run_comprehensive_experiments.py
# skips whatever a step's checkpoint.json already marks done.
#
# Progress, checkable without touching the running process:
#   cat experiment_results/RUNNING.txt      # current step
#   tail -f logs/cross_run_<timestamp>.log  # live log (this launch)
#   cat experiment_results/STATUS.md        # per-project rollup
#   cat experiment_results/all_projects_results.csv

set -uo pipefail

# Prevent two copies running at once - held for the entire run, released
# automatically on exit, crash, or kill. Deliberately a DIFFERENT lock
# file from exp3.1's run_full_experiment.sh
# (/tmp/simrgl_exp3.1_run_full_experiment.lock) so the two experiments'
# launch scripts never block each other - they still shouldn't run
# concurrently in practice (shared GPU/Postgres), but that's an
# operational decision made by whoever launches them, not something this
# lock should enforce.
LOCK_FILE="/tmp/simrgl_exp3.2_run_cross_experiment.lock"
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
  echo "Another run_cross_experiment.sh is already running (lock: $LOCK_FILE) - not starting a second copy."
  echo "Check with: pgrep -af run_cross_experiment.sh"
  exit 1
fi

MODELS="bge-small bge-large"
STRATEGIES="recent modn"
TARGETS="file module"
WINDOWS="w100 w1000 all"
TRAIN_SOURCES="title desc diff"
QUERY_SOURCES="title desc comments"

# project:test_size, ordered smallest-task-count-first (project_stats.csv,
# same ordering exp3.1 used) so an interruption early on still leaves the
# broadest possible partial dataset. No agilebill (no TASK rows - no query
# side, categorically out of scope for this experiment, see README.md).
STEPS=(
  "celery:200"
  "rubocop:200"
  "pulumi:200"
  "sonar:200"
  "flink:200"
  "hadoop:200"
  "spark:200"
  "kubernetes:200"
  "vscode:200"
)

mkdir -p logs experiment_results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/cross_run_${TIMESTAMP}.log"
STATUS_FILE="experiment_results/RUNNING.txt"

log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG_FILE"
}

TOTAL_STEPS=${#STEPS[@]}
log "===== Cross-vocabulary experiment run starting - log: $LOG_FILE ====="
log "Steps ($TOTAL_STEPS): ${STEPS[*]}"
log "Strategies: $STRATEGIES | Models: $MODELS | Targets: $TARGETS | Windows: $WINDOWS"
log "Train sources: $TRAIN_SOURCES | Query sources: $QUERY_SOURCES"

log "----- Ensuring Postgres is up (./run_postgres.sh) -----"
if ! ./run_postgres.sh 2>&1 | tee -a "$LOG_FILE"; then
  log "!!!!! Postgres did not come up - aborting before wasting GPU time on doomed ETL runs !!!!!"
  exit 1
fi

PREV_PROJECT="none"
STEP_NUM=0

for step in "${STEPS[@]}"; do
  STEP_NUM=$((STEP_NUM + 1))
  IFS=':' read -r PROJECT TEST_SIZE <<< "$step"

  echo "$(date -Iseconds) step ${STEP_NUM}/${TOTAL_STEPS}: ${PROJECT}/cross (prev=${PREV_PROJECT})" > "$STATUS_FILE"

  log "----- [${STEP_NUM}/${TOTAL_STEPS}] Switching to ${PROJECT} (task_unit=cross) -----"
  ./switch_project.sh --from "$PREV_PROJECT" --to "$PROJECT" --task-unit cross --yes 2>&1 | tee -a "$LOG_FILE"

  log "----- [${STEP_NUM}/${TOTAL_STEPS}] Running ${PROJECT}/cross: train_sources=[$TRAIN_SOURCES] query_sources=[$QUERY_SOURCES] test_size=$TEST_SIZE -----"
  python3.12 run_comprehensive_experiments.py \
    --project "$PROJECT" --task-unit cross \
    --models $MODELS --strategies $STRATEGIES \
    --train-sources $TRAIN_SOURCES --query-sources $QUERY_SOURCES \
    --targets $TARGETS --windows $WINDOWS \
    --test-size "$TEST_SIZE" \
    --yes 2>&1 | tee -a "$LOG_FILE"

  RC=${PIPESTATUS[0]}
  if [ "$RC" -ne 0 ]; then
    log "!!!!! ${PROJECT}/cross exited with code ${RC} - continuing to next step anyway !!!!!"
  else
    log "----- [${STEP_NUM}/${TOTAL_STEPS}] ${PROJECT}/cross finished -----"
  fi

  python3.12 status.py 2>&1 | tee -a "$LOG_FILE"
  python3.12 aggregate_results.py 2>&1 | tee -a "$LOG_FILE"

  PREV_PROJECT="$PROJECT"
done

echo "$(date -Iseconds) all ${TOTAL_STEPS} steps complete" > "$STATUS_FILE"
log "===== Cross-vocabulary experiment run complete ====="
