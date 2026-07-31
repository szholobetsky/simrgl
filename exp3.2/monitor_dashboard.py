#!/usr/bin/env python3
"""
Three-level progress dashboard for run_cross_experiment.sh (exp3.2).

Level 1 - Overall: eval-runs completed across all 9 project steps.
Level 2 - Current step: eval-runs completed within the project currently running.
Level 3 - Current activity: live sub-progress of whatever's running right now
           (embedding batch, upsert batch, or eval query loop), parsed from
           the latest tqdm line in the run's log file.

No third-party dependencies - stdlib only, runs with any python3.

Usage:
    python3 monitor_dashboard.py            # print once
    python3 monitor_dashboard.py --watch    # refresh every 10s (Ctrl-C to stop)
"""

import glob
import json
import os
import re
import sys
import time

RESULTS_DIR = 'experiment_results'
LOG_GLOB = 'logs/cross_run_*.log'

# Fixed grid in run_cross_experiment.sh: 1 model (bge-small only - see
# EXPERIMENT_PLAN.md's model-scope note) x 2 strategies x 3 train_sources
# x 3 query_sources x 2 targets x 3 windows = 108 eval-runs per project
# (same count for every step - single task_unit='cross' pass, no
# ticket/commit alternation, no agilebill).
VARIANTS_PER_STEP = 1 * 2 * 3 * 3 * 2 * 3
TOTAL_STEPS = 9


def load_checkpoint(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def bar(fraction, width=40):
    fraction = max(0.0, min(1.0, fraction if fraction == fraction else 0.0))  # NaN guard
    filled = int(round(fraction * width))
    return '[' + '#' * filled + '-' * (width - filled) + f'] {fraction * 100:5.1f}%'


def get_all_checkpoints():
    pattern = os.path.join(RESULTS_DIR, '*', '*', 'checkpoint.json')
    result = {}
    for path in glob.glob(pattern):
        parts = path.split(os.sep)
        task_unit = parts[-2]
        project = parts[-3]
        data = load_checkpoint(path)
        if data:
            result[(project, task_unit)] = data
    return result


def get_current_step():
    running_file = os.path.join(RESULTS_DIR, 'RUNNING.txt')
    if not os.path.exists(running_file):
        return None
    with open(running_file) as f:
        line = f.read().strip()
    m = re.search(r'step (\d+)/(\d+): (\S+)/(\S+)', line)
    if not m:
        return None
    step_num, total, project, task_unit = m.groups()
    return {'step_num': int(step_num), 'total': int(total),
             'project': project, 'task_unit': task_unit, 'raw': line}


def get_latest_log():
    logs = sorted(glob.glob(LOG_GLOB), key=os.path.getmtime)
    return logs[-1] if logs else None


def get_current_variant(log_path, tail_bytes=20000):
    """
    Parse the most recent "Processing variant: <model>_<strategy>_<source>_
    <target>_<window>" line - printed by _process_source_window() /
    run_experiment_variant() in run_comprehensive_experiments.py before
    every variant. This is the direct answer to "which window/strategy/
    source is being tested right now" - model/strategy/source/target/window
    never contain underscores themselves, so splitting on '_' cleanly
    gives 5 parts.
    """
    if not log_path or not os.path.exists(log_path):
        return None
    with open(log_path, 'rb') as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - tail_bytes))
        chunk = f.read().decode('utf-8', errors='ignore')

    matches = re.findall(r'Processing variant:\s*(\S+)', chunk)
    if not matches:
        return None
    parts = matches[-1].split('_')
    if len(parts) != 5:
        return {'raw': matches[-1]}
    model, strategy, source, target, window = parts
    return {'raw': matches[-1], 'model': model, 'strategy': strategy,
            'source': source, 'target': target, 'window': window}


def get_live_activity(log_path, tail_bytes=8000):
    """Parse the most recent tqdm progress snapshot out of the log tail.
    tqdm writes \\r-separated updates when not attached to a TTY (as here,
    piped through tee) - they land as one visually-run-together chunk."""
    if not log_path or not os.path.exists(log_path):
        return None
    with open(log_path, 'rb') as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - tail_bytes))
        chunk = f.read().decode('utf-8', errors='ignore')

    for piece in reversed(re.split(r'[\r\n]', chunk)):
        m = re.search(r'^(.*?):\s*(\d+)%\|.*?\|\s*(\d+)/(\d+)', piece.strip())
        if m:
            label, pct, done, total = m.groups()
            return {'label': label.strip(), 'pct': int(pct), 'done': int(done), 'total': int(total)}
    return None


def render():
    checkpoints = get_all_checkpoints()
    current = get_current_step()

    overall_done = sum(len(cp.get('completed_experiments', [])) for cp in checkpoints.values())
    overall_total = TOTAL_STEPS * VARIANTS_PER_STEP
    overall_failed = sum(
        len(cp.get('failed_etl', [])) + len(cp.get('failed_experiments', []))
        for cp in checkpoints.values()
    )

    lines = []
    lines.append("=" * 70)
    lines.append("exp3.2 CROSS-VOCABULARY EXPERIMENT DASHBOARD")
    lines.append("=" * 70)
    lines.append("")

    steps_fully_done = sum(
        1 for cp in checkpoints.values()
        if len(cp.get('completed_experiments', [])) >= VARIANTS_PER_STEP
        and not cp.get('failed_etl') and not cp.get('failed_experiments')
    )
    lines.append(f"Level 1 - Overall experiment: {overall_done}/{overall_total} variant-runs done "
                 f"({steps_fully_done} of {TOTAL_STEPS} steps fully finished, {overall_failed} failed)")
    lines.append(bar(overall_done / overall_total if overall_total else 0))
    lines.append("")

    if current:
        key = (current['project'], current['task_unit'])
        cp = checkpoints.get(key)
        step_done = len(cp.get('completed_experiments', [])) if cp else 0
        step_failed = (len(cp.get('failed_etl', [])) + len(cp.get('failed_experiments', []))) if cp else 0
        lines.append(f"Level 2 - Current project ({current['step_num']}/{current['total']}): "
                     f"{current['project']}/{current['task_unit']} "
                     f"({step_done}/{VARIANTS_PER_STEP} variants, {step_failed} failed)")
        lines.append(bar(step_done / VARIANTS_PER_STEP))
    else:
        lines.append("Level 2 - Current project: (not started yet)")
        lines.append(bar(0))
    lines.append("")

    log_path = get_latest_log()

    variant = get_current_variant(log_path)
    if variant and 'model' in variant:
        lines.append(f"Current variant: model={variant['model']}  strategy={variant['strategy']}  "
                     f"source={variant['source']}  target={variant['target']}  window={variant['window']}")
    elif variant:
        lines.append(f"Current variant: {variant['raw']}")
    else:
        lines.append("Current variant: (none seen yet in this log)")
    lines.append("")

    activity = get_live_activity(log_path)
    if activity:
        lines.append(f"Level 3 - Current step's live task: {activity['label']} ({activity['done']}/{activity['total']})")
        lines.append(bar(activity['pct'] / 100))
    else:
        lines.append("Level 3 - Current step's live task: (idle / between variants)")
        lines.append(bar(0))
    lines.append("")

    lines.append("-" * 70)
    lines.append(f"Log:     {log_path or '(none yet)'}")
    lines.append(f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    return '\n'.join(lines)


def parse_interval(argv, default=10):
    for i, arg in enumerate(argv):
        if arg == '--interval' and i + 1 < len(argv):
            try:
                return max(1, int(argv[i + 1]))
            except ValueError:
                pass
        if arg.startswith('--interval='):
            try:
                return max(1, int(arg.split('=', 1)[1]))
            except ValueError:
                pass
    return default


def main():
    watch = '--watch' in sys.argv or '-w' in sys.argv
    interval = parse_interval(sys.argv)

    if not watch:
        print(render())
        return

    try:
        while True:
            os.system('clear')
            print(render())
            print(f"\n(refreshing every {interval}s - Ctrl-C to stop)")
            time.sleep(interval)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
