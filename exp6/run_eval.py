"""
run_eval.py

Evaluates all three RQs against every project in data/projects.csv:
  RQ1 — per-mode baseline (task/file/aggr) + Wilcoxon significance between modes
  RQ2 — combination strategies: Intersection/Union/Weighted/Borda (combiners.py)
        + simargl's own RRF (`simargl rrf`)
  RQ3 — multi-granularity: (a) module/extension prediction quality per mode,
        (b) prior-filtering lambda-grid re-ranking of file results

RQ2 evaluates every non-empty subset of size >= 2 of {task, file, aggr} —
{task,file}, {task,aggr}, {file,aggr}, {task,file,aggr} — under each of the
5 combination strategies (Intersection/Union/Weighted/Borda/RRF). Pairs
matter methodologically, not just the all-three combo: `file` and `aggr`
both search the same files.int8 index (only the query vector differs — raw
embedding vs score-weighted task centroid), so file+aggr tests whether aggr
adds anything beyond direct file search; `task` uses a structurally
different signal (historical co-change, not code semantics), so
task+file/task+aggr are the more likely candidates for a real gain over the
best single mode. See EXPERIMENT_PLANNING.md for the full hypothesis per RQ.

retriever.py calls simargl's Python API in-process (not the CLI via
subprocess) — the embedder loads once per run of this script, not once per
call (see retriever.py docstring for the measured ~180x cost difference).
Each test task still triggers one `search()` call per mode (task/file/aggr,
shared across RQ1/RQ2/RQ3) + one `rrf_search()` call per RQ2 subset (4
subsets = 4 rrf calls, since RRF's merge is simargl's own algorithm, not
reimplemented here) — these are just function calls now, not process
spawns. The 4 non-RRF combiners per subset are pure in-memory re-ranking of
the single shared mode fetch — see IMPLEMENTATION_DETAIL_CL.md.

Must run AFTER scripts/prepare_units_db.py and scripts/prep_index.py.
"""
from __future__ import annotations

import argparse
import csv as csv_module
import json
import os
from itertools import combinations
from typing import Dict, List

import config
import combiners
import granularity
import utils
from retriever import SimarglRetriever

logger = utils.logger

MODES = ('task', 'file', 'aggr')

# All non-empty subsets of size >= 2 -- {task,file}, {task,aggr}, {file,aggr}, {task,file,aggr}.
# Size-1 "subsets" are just the RQ1 baseline, already in mode_comparison.csv.
MODE_SUBSETS = [subset for r in (2, 3) for subset in combinations(MODES, r)]
STRATEGY_NAMES = list(combiners.COMBINERS.keys()) + ['rrf']


def load_test_set(project: str) -> List[dict]:
    path = config.test_set_path(project)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run scripts/prepare_units_db.py first.")
    with open(path, encoding='utf-8') as f:
        test_set = json.load(f)
    if not utils.validate_test_set(test_set):
        raise ValueError(f"Test set validation failed for project '{project}': {path}")
    return test_set


def _query_text(task: dict) -> str:
    return utils.combine_text_fields(task, ['TITLE', 'DESCRIPTION'])


def fetch_mode_data(retriever: SimarglRetriever, test_set: List[dict]) -> List[dict]:
    """One search() call per mode per task — the single fetch shared by
    RQ1/RQ2/RQ3 below, so no mode is ever re-queried."""
    records = []
    for task in test_set:
        query = _query_text(task)
        mode_data = {mode: retriever.search(query, mode) for mode in MODES}
        records.append({'task': task, 'mode_data': mode_data})
    return records


# --------------------------------------------------------------------- RQ1

def evaluate_rq1(records: List[dict], k_values: List[int]):
    per_task_metrics = {mode: [] for mode in MODES}
    for rec in records:
        gt = set(rec['task']['relevant_files'])
        for mode in MODES:
            paths = [f['path'] for f in rec['mode_data'][mode].get('files', [])]
            per_task_metrics[mode].append(utils.calculate_metrics_for_query(paths, gt, k_values))
    aggregated = {mode: utils.aggregate_metrics(per_task_metrics[mode], k_values) for mode in MODES}
    return aggregated, per_task_metrics


def wilcoxon_significance(per_task_metrics: Dict[str, list]) -> List[dict]:
    from scipy.stats import wilcoxon
    modes = list(per_task_metrics.keys())
    rows = []
    for i in range(len(modes)):
        for j in range(i + 1, len(modes)):
            m1, m2 = modes[i], modes[j]
            ap1 = [m['AP'] for m in per_task_metrics[m1]]
            ap2 = [m['AP'] for m in per_task_metrics[m2]]
            try:
                stat, p = wilcoxon(ap1, ap2)
            except ValueError:
                # all differences are zero, or too few non-zero pairs — scipy raises
                stat, p = float('nan'), float('nan')
            rows.append({'mode_a': m1, 'mode_b': m2, 'statistic': stat, 'p_value': p})
    return rows


# --------------------------------------------------------------------- RQ2

def evaluate_rq2(retriever: SimarglRetriever, records: List[dict], k_values: List[int],
                 mode_weights: Dict[str, float], project: str) -> List[dict]:
    """Every mode subset (>= 2 modes) x every strategy -- see module docstring."""
    per_key_metrics = {(subset, strat): [] for subset in MODE_SUBSETS for strat in STRATEGY_NAMES}

    for rec in records:
        task, mode_data = rec['task'], rec['mode_data']
        gt = set(task['relevant_files'])
        mode_paths = {mode: [f['path'] for f in mode_data[mode].get('files', [])] for mode in MODES}
        query = _query_text(task)

        for subset in MODE_SUBSETS:
            subset_paths = {m: mode_paths[m] for m in subset}

            for strategy, fn in combiners.COMBINERS.items():
                retrieved = fn(subset_paths, mode_weights=mode_weights)
                per_key_metrics[(subset, strategy)].append(
                    utils.calculate_metrics_for_query(retrieved, gt, k_values)
                )

            rrf_sources = ",".join(f"{m}:{project}" for m in subset)
            rrf_retrieved = retriever.rrf_search(query, rrf_sources)
            per_key_metrics[(subset, 'rrf')].append(
                utils.calculate_metrics_for_query(rrf_retrieved, gt, k_values)
            )

    rows = []
    for (subset, strategy), metrics_list in per_key_metrics.items():
        agg = utils.aggregate_metrics(metrics_list, k_values)
        rows.append({'subset': '+'.join(subset), 'strategy': strategy, **agg})
    return rows


# --------------------------------------------------------------------- RQ3

def evaluate_rq3_prior_filtering(records: List[dict], k_values: List[int]) -> List[dict]:
    grid_metrics = {
        (mode, lm, le): []
        for mode in MODES for lm in config.LAMBDA_GRID for le in config.LAMBDA_GRID
    }

    for rec in records:
        task, mode_data = rec['task'], rec['mode_data']
        gt = set(task['relevant_files'])

        for mode in MODES:
            data = mode_data[mode]
            files = data.get('files', [])
            top_modules = [m['module'] for m in data.get('modules', [])[:config.TOP_M_MODULES]]
            top_exts = [e['extension'] for e in granularity.top_extensions(files, config.TOP_M_EXTENSIONS)]

            for lm in config.LAMBDA_GRID:
                for le in config.LAMBDA_GRID:
                    retrieved = granularity.apply_prior_filter(files, top_modules, top_exts, lm, le)
                    grid_metrics[(mode, lm, le)].append(
                        utils.calculate_metrics_for_query(retrieved, gt, k_values)
                    )

    rows = []
    for (mode, lm, le), metrics_list in grid_metrics.items():
        agg = utils.aggregate_metrics(metrics_list, k_values)
        rows.append({'mode': mode, 'lambda_module': lm, 'lambda_ext': le, **agg})
    return rows


def evaluate_rq3_multi_granularity(records: List[dict], k_values: List[int]) -> List[dict]:
    """README RQ3 steps 1-2: which mode predicts modules/extensions best,
    independent of the prior-filtering grid above."""
    keys = [(mode, gran) for mode in MODES for gran in ('module', 'extension')]
    per_key_metrics = {k: [] for k in keys}

    for rec in records:
        task, mode_data = rec['task'], rec['mode_data']
        gt_modules = {utils.extract_module_path(f) for f in task['relevant_files']}
        gt_exts = {granularity.extract_extension(f) for f in task['relevant_files']}

        for mode in MODES:
            data = mode_data[mode]
            predicted_modules = [m['module'] for m in data.get('modules', [])]
            predicted_exts = [
                e['extension'] for e in granularity.top_extensions(data.get('files', []), top_m=20)
            ]
            per_key_metrics[(mode, 'module')].append(
                utils.calculate_metrics_for_query(predicted_modules, gt_modules, k_values)
            )
            per_key_metrics[(mode, 'extension')].append(
                utils.calculate_metrics_for_query(predicted_exts, gt_exts, k_values)
            )

    rows = []
    for (mode, gran), metrics_list in per_key_metrics.items():
        agg = utils.aggregate_metrics(metrics_list, k_values)
        rows.append({'mode': mode, 'granularity': gran, **agg})
    return rows


# --------------------------------------------------------------------- driver

def run_project(project: dict, k_values: List[int]):
    name = project['name']
    logger.info("=" * 80)
    logger.info(f"Evaluating project: {name}")

    test_set = load_test_set(name)
    retriever = SimarglRetriever(project_id=name)

    records = fetch_mode_data(retriever, test_set)
    logger.info(f"Fetched mode data for {len(records)} tasks x {len(MODES)} modes")

    rq1_agg, rq1_per_task = evaluate_rq1(records, k_values)
    rq1_rows = [{'project': name, 'mode': mode, **metrics} for mode, metrics in rq1_agg.items()]
    significance_rows = [{'project': name, **row} for row in wilcoxon_significance(rq1_per_task)]
    mode_weights = {mode: metrics['MAP'] for mode, metrics in rq1_agg.items()}

    rq2_rows = [{'project': name, **row} for row in evaluate_rq2(retriever, records, k_values, mode_weights, name)]

    rq3_rows = [{'project': name, **row} for row in evaluate_rq3_prior_filtering(records, k_values)]
    multigran_rows = [{'project': name, **row} for row in evaluate_rq3_multi_granularity(records, k_values)]

    return rq1_rows, significance_rows, rq2_rows, rq3_rows, multigran_rows


def write_csv(rows: List[dict], path: str):
    if not rows:
        logger.warning(f"No rows to write for {path}")
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv_module.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {len(rows)} rows to {path}")


def main():
    parser = argparse.ArgumentParser(description="Run exp6 evaluation (RQ1 + RQ2 + RQ3)")
    parser.add_argument('--project', default=None, help="Limit to one project name from projects.csv")
    args = parser.parse_args()

    projects = config.load_projects()
    if args.project:
        projects = [p for p in projects if p['name'] == args.project]
        if not projects:
            raise SystemExit(f"Project '{args.project}' not found in {config.PROJECTS_MANIFEST}")

    all_rq1, all_sig, all_rq2, all_rq3, all_multigran = [], [], [], [], []
    for project in projects:
        rq1_rows, sig_rows, rq2_rows, rq3_rows, multigran_rows = run_project(project, config.TOP_K_VALUES)
        all_rq1 += rq1_rows
        all_sig += sig_rows
        all_rq2 += rq2_rows
        all_rq3 += rq3_rows
        all_multigran += multigran_rows

    write_csv(all_rq1, os.path.join(config.RESULTS_DIR, 'mode_comparison.csv'))
    write_csv(all_sig, os.path.join(config.RESULTS_DIR, 'mode_significance.csv'))
    write_csv(all_rq2, os.path.join(config.RESULTS_DIR, 'combination_strategies.csv'))
    write_csv(all_rq3, os.path.join(config.RESULTS_DIR, 'prior_filtering.csv'))
    write_csv(all_multigran, os.path.join(config.RESULTS_DIR, 'multi_granularity.csv'))

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
