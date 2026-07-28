#!/bin/bash
#
# Full experiment: the complete combinatorial grid, no shortcuts, no
# subsetting decided from a single project's numbers.
#
#   10 projects x 2 task_unit criteria (ticket, commit) x 2 split
#   strategies (recent, modn) x 2 embedding models (bge-small, bge-large -
#   the only two that fit this GPU's 6GB VRAM, the rest OOM) x 18
#   (source x target x window) combinations per task_unit.
#
# Two deliberate exceptions, both explicit rather than silently decided:
#   - agilebill: has ZERO Jira/tracker data (no TASK rows at all) - ticket
#     mode is not just weak there, it's impossible. Runs commit-mode only.
#   - agilebill: only 120 total commits - the usual 200-task held-out test
#     set would leave nothing to train on. Uses --test-size 20 instead.
#     We're running it anyway (not skipping it) specifically to see how
#     bad the numbers get at this volume - that's itself a data point.
#
# Source variants differ by task_unit:
#   - ticket:  title / desc / comments   (TITLE, TITLE+DESCRIPTION, TITLE+DESCRIPTION+COMMENTS)
#   - commit:  title / desc / diff       (subject line, full message, full message+diff)
#   COMMENTS is always '' for a raw commit, so a 'comments' variant would
#   be byte-identical to 'desc' in commit-mode - 'diff' (the commit's
#   actual changed-code content, capped at 4000 chars/task - embedding
#   models truncate long inputs anyway) fills that "extra noisy context"
#   role instead, and is a more informative test than a guaranteed no-op.
#
# Meant to run for a long time (days, plausibly longer) - launch it
# detached in its own session, not as a subprocess of a chat session:
#   nohup ./run_full_experiment.sh > /dev/null 2>&1 &
#   disown
# or inside tmux/screen.
#
# Fully resumable from a cold start at any point (power loss, reboot,
# anything) - just run this exact same command again. Every step is
# idempotent: run_comprehensive_experiments.py skips whatever a step's
# checkpoint.json already marks done and only computes what's missing, so
# re-running from the top after an interruption re-verifies already-done
# steps quickly (checkpoint lookups, not recomputation) and then continues
# real work exactly where it stopped. No manual bookkeeping needed.
#
# Progress, checkable without touching the running process:
#   cat experiment_results/RUNNING.txt      # current step
#   tail -f logs/full_run_<timestamp>.log   # live log (this launch)
#   cat experiment_results/STATUS.md        # per-(project,task_unit) rollup
#   cat experiment_results/all_projects_results.csv

set -uo pipefail

# Prevent two copies running at once (e.g. manually launched twice by
# mistake) - held for the entire run, released automatically on exit,
# crash, or kill. No stale-lockfile problem: flock ties the lock to this
# process's open file descriptor, not to the file's mere existence.
LOCK_FILE="/tmp/simrgl_exp3.1_run_full_experiment.lock"
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
  echo "Another run_full_experiment.sh is already running (lock: $LOCK_FILE) - not starting a second copy."
  echo "Check with: pgrep -af run_full_experiment.sh"
  exit 1
fi

MODELS="bge-small bge-large"
STRATEGIES="recent modn"
TARGETS="file module"
WINDOWS="w100 w1000 all"

SOURCES_TICKET="title desc comments"
SOURCES_COMMIT="title desc diff"

# project:task_unit:test_size, ordered smallest-task-count-first
# (db/project_stats.csv) so an interruption early on still leaves the
# broadest possible partial dataset. ticket+commit run back-to-back per
# project so each project's full ticket-vs-commit comparison lands
# together. agilebill last, commit-only, tiny test_size (see header).
STEPS=(
  "celery:ticket:200"
  "celery:commit:200"
  "rubocop:ticket:200"
  "rubocop:commit:200"
  "pulumi:ticket:200"
  "pulumi:commit:200"
  "sonar:ticket:200"
  "sonar:commit:200"
  "flink:ticket:200"
  "flink:commit:200"
  "hadoop:ticket:200"
  "hadoop:commit:200"
  "spark:ticket:200"
  "spark:commit:200"
  "kubernetes:ticket:200"
  "kubernetes:commit:200"
  "vscode:ticket:200"
  "vscode:commit:200"
  "agilebill:commit:20"
)

mkdir -p logs experiment_results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/full_run_${TIMESTAMP}.log"
STATUS_FILE="experiment_results/RUNNING.txt"

log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG_FILE"
}

TOTAL_STEPS=${#STEPS[@]}
log "===== Full experiment run starting - log: $LOG_FILE ====="
log "Steps ($TOTAL_STEPS): ${STEPS[*]}"
log "Strategies: $STRATEGIES | Models: $MODELS | Targets: $TARGETS | Windows: $WINDOWS"

log "----- Ensuring Postgres is up (./run_postgres.sh) -----"
if ! ./run_postgres.sh 2>&1 | tee -a "$LOG_FILE"; then
  log "!!!!! Postgres did not come up - aborting before wasting GPU time on doomed ETL runs !!!!!"
  exit 1
fi

PREV_PROJECT="none"
PREV_TASK_UNIT="ticket"
STEP_NUM=0

for step in "${STEPS[@]}"; do
  STEP_NUM=$((STEP_NUM + 1))
  IFS=':' read -r PROJECT TASK_UNIT TEST_SIZE <<< "$step"

  if [ "$TASK_UNIT" = "commit" ]; then
    SOURCES="$SOURCES_COMMIT"
  else
    SOURCES="$SOURCES_TICKET"
  fi

  echo "$(date -Iseconds) step ${STEP_NUM}/${TOTAL_STEPS}: ${PROJECT}/${TASK_UNIT} (prev=${PREV_PROJECT}/${PREV_TASK_UNIT})" > "$STATUS_FILE"

  log "----- [${STEP_NUM}/${TOTAL_STEPS}] Switching to ${PROJECT} (task_unit=${TASK_UNIT}) -----"
  ./switch_project.sh --from "$PREV_PROJECT" --to "$PROJECT" --task-unit "$PREV_TASK_UNIT" --yes 2>&1 | tee -a "$LOG_FILE"

  log "----- [${STEP_NUM}/${TOTAL_STEPS}] Running ${PROJECT}/${TASK_UNIT}: sources=[$SOURCES] test_size=$TEST_SIZE -----"
  python3.12 run_comprehensive_experiments.py \
    --project "$PROJECT" --task-unit "$TASK_UNIT" \
    --models $MODELS --strategies $STRATEGIES \
    --sources $SOURCES --targets $TARGETS --windows $WINDOWS \
    --test-size "$TEST_SIZE" \
    --yes 2>&1 | tee -a "$LOG_FILE"

  RC=${PIPESTATUS[0]}
  if [ "$RC" -ne 0 ]; then
    log "!!!!! ${PROJECT}/${TASK_UNIT} exited with code ${RC} - continuing to next step anyway !!!!!"
  else
    log "----- [${STEP_NUM}/${TOTAL_STEPS}] ${PROJECT}/${TASK_UNIT} finished -----"
  fi

  python3.12 status.py 2>&1 | tee -a "$LOG_FILE"
  python3.12 aggregate_results.py 2>&1 | tee -a "$LOG_FILE"

  PREV_PROJECT="$PROJECT"
  PREV_TASK_UNIT="$TASK_UNIT"
done

echo "$(date -Iseconds) all ${TOTAL_STEPS} steps complete" > "$STATUS_FILE"
log "===== Full experiment run complete ====="
