"""
scripts/prepare_units_db.py

For each project in data/projects.csv:
  1. Load TASK + COMMITS/RAWDATA from db_path (exp3.1/exp3.2 schema — see
     IMPLEMENTATION_DETAIL_CL.md section 2: columns already match what
     simargl's indexer expects, only the task-table name differs).
  2. Chronological train/test split (recent | modn) — same logic as
     exp3.2/etl_pipeline.py ETLPipeline.create_split().
  3. Optional time window filter on training tasks (WINDOW_VARIANTS).
  4. Write a NEW sqlite db containing ONLY train tasks, with tables named
     TASKS / COMMITS (what simargl.indexer.index_units expects) —
     exp6/data/db/<project>_<split>_train.db. This is the leakage-prevention
     step, not a schema converter (IMPLEMENTATION_DETAIL_CL.md section 3):
     without it, `simargl index units` would contain every test task too,
     and mode=task would trivially find each test task via itself.
  5. Build ground-truth test_set.json from the FULL commits table restricted
     to test task names — exp6/data/test_set_<project>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# exp6/ (parent of scripts/) holds config.py and utils.py — not on sys.path
# when this file is launched directly as `python scripts/prepare_units_db.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from utils import extract_file_path, logger


def get_commits_table_name(conn: sqlite3.Connection) -> str:
    """Detect the commits table name — RAWDATA (older CodeXplorer DBs) or
    COMMITS (newer ones, see exp3.1/project_stats.csv). Same columns either
    way. Copied from exp3.2/utils.py:get_commits_table_name."""
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('COMMITS', 'RAWDATA')"
    )
    tables = {row[0] for row in cur.fetchall()}
    if 'COMMITS' in tables:
        return 'COMMITS'
    if 'RAWDATA' in tables:
        return 'RAWDATA'
    raise ValueError("Database has neither a COMMITS nor a RAWDATA table")


def load_data(db_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(db_path)
    commits_table = get_commits_table_name(conn)
    commits_df = pd.read_sql_query(
        f"SELECT SHA, PATH, MESSAGE, CMT_DATE, TASK_NAME FROM {commits_table}", conn
    )
    tasks_df = pd.read_sql_query(
        "SELECT ID, NAME, TITLE, DESCRIPTION, COMMENTS FROM TASK", conn
    )
    conn.close()
    logger.info(f"Loaded {len(tasks_df)} tasks and {len(commits_df)} commit rows from {db_path}")
    return tasks_df, commits_df


def create_split(
    tasks_df: pd.DataFrame, split_strategy: str, test_size: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verbatim logic from exp3.2/etl_pipeline.py:174-214 (ETLPipeline.create_split)."""
    tasks_df = tasks_df.sort_values('ID').reset_index(drop=True)
    total_tasks = len(tasks_df)
    indices = np.arange(total_tasks)

    logger.info(f"Creating split using strategy: {split_strategy}")
    logger.info(f"Total tasks: {total_tasks}, Test size: {test_size}")

    if split_strategy == 'recent':
        test_indices = indices[-test_size:]
        train_indices = indices[:-test_size]
    elif split_strategy == 'modn':
        step = total_tasks // test_size
        test_indices = indices[::step][:test_size]
        train_indices = np.setdiff1d(indices, test_indices)
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")

    train_tasks = tasks_df.iloc[train_indices].copy()
    test_tasks = tasks_df.iloc[test_indices].copy()

    logger.info(f"Train tasks: {len(train_tasks)}, Test tasks: {len(test_tasks)}")
    return train_tasks, test_tasks


def apply_time_window(train_tasks: pd.DataFrame, window_size: int | None) -> pd.DataFrame:
    """Verbatim logic from exp3.2/etl_pipeline.py:216-250 (recent-tail truncation)."""
    if window_size is None:
        logger.info("Using ALL training tasks (no window limit)")
        return train_tasks
    windowed = train_tasks.iloc[-window_size:]
    logger.info(f"Window filtered: {len(windowed)} tasks (from {len(train_tasks)})")
    return windowed


def prepare_test_set(test_tasks: pd.DataFrame, commits_df: pd.DataFrame) -> list[dict]:
    """Ground truth per test task — verbatim logic from
    exp3.2/etl_pipeline.py:252-287 (ETLPipeline.prepare_test_set), built from
    the FULL commits table, not the train-only subset."""
    test_set = []
    for _, task in test_tasks.iterrows():
        relevant_files = (
            commits_df[commits_df['TASK_NAME'] == task['NAME']]['PATH'].unique().tolist()
        )
        relevant_files = [extract_file_path(f) for f in relevant_files]
        relevant_files = [f for f in relevant_files if f != "unknown"]
        test_set.append({
            'NAME': task['NAME'],
            'TITLE': task.get('TITLE', ''),
            'DESCRIPTION': task.get('DESCRIPTION', ''),
            'COMMENTS': task.get('COMMENTS', ''),
            'relevant_files': relevant_files,
        })
    logger.info(f"Prepared test set with {len(test_set)} tasks")
    return test_set


def write_train_db(train_tasks: pd.DataFrame, commits_df: pd.DataFrame, out_path: str) -> None:
    """Materialize a train-only sqlite db with TASKS/COMMITS tables — the
    schema `simargl index units` expects. Doing the leakage-exclusion filter
    and the table rename in the same write is deliberate: it's one
    materialization step either way, not a separate "converter" pass."""
    train_names = set(train_tasks['NAME'])
    train_commits = commits_df[commits_df['TASK_NAME'].isin(train_names)]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)

    conn = sqlite3.connect(out_path)
    train_tasks[['ID', 'NAME', 'TITLE', 'DESCRIPTION']].to_sql('TASKS', conn, index=False)
    train_commits[['SHA', 'PATH', 'TASK_NAME']].to_sql('COMMITS', conn, index=False)
    conn.execute("CREATE INDEX idx_commits_task_name ON COMMITS(TASK_NAME)")
    conn.commit()
    conn.close()

    logger.info(
        f"Wrote train-only DB: {out_path} "
        f"(TASKS={len(train_tasks)}, COMMITS={len(train_commits)})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build a leakage-safe train-only units DB + test_set.json per project"
    )
    parser.add_argument(
        '--split', choices=list(config.SPLIT_STRATEGIES), default='modn',
        help="modn is the honest baseline (AGENTS.md locked finding #2); "
             "recent inflates MAP via temporal proximity — default is modn on purpose"
    )
    parser.add_argument('--test-size', type=int, default=config.TEST_SIZE)
    parser.add_argument('--window', choices=list(config.WINDOW_VARIANTS), default='all')
    parser.add_argument('--project', default=None, help="Limit to one project name from projects.csv")
    args = parser.parse_args()

    projects = config.load_projects()
    if args.project:
        projects = [p for p in projects if p['name'] == args.project]
        if not projects:
            raise SystemExit(f"Project '{args.project}' not found in {config.PROJECTS_MANIFEST}")

    window_size = config.WINDOW_VARIANTS[args.window]['size']

    for proj in projects:
        name, db_path = proj['name'], proj['db_path']
        logger.info("=" * 80)
        logger.info(f"Project: {name}  (db={db_path})")

        tasks_df, commits_df = load_data(db_path)
        train_tasks, test_tasks = create_split(tasks_df, args.split, args.test_size)
        train_tasks = apply_time_window(train_tasks, window_size)

        test_set = prepare_test_set(test_tasks, commits_df)
        out_test_set = config.test_set_path(name)
        os.makedirs(os.path.dirname(out_test_set) or '.', exist_ok=True)
        with open(out_test_set, 'w', encoding='utf-8') as f:
            json.dump(test_set, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved test set to {out_test_set}")

        out_train_db = config.train_db_path(name, args.split)
        write_train_db(train_tasks, commits_df, out_train_db)


if __name__ == '__main__':
    main()
