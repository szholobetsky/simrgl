"""
scripts/prep_index.py

For each project in data/projects.csv:
  1. `simargl index files <repo_path>` — current HEAD of the repo checkout
     (MVP decision, see IMPLEMENTATION_DETAIL_CL.md section 7: no
     point-in-time checkout yet, tracked as backlog item #1 in section 10).
  2. `simargl index units <train_db>` — the LEAKAGE-SAFE train-only DB
     produced by prepare_units_db.py, never the raw project DB.
  3. `simargl status --project <name>` — fail fast if either index is
     empty, instead of discovering it mid-eval (mode=aggr needs BOTH
     indexes, searcher.py:139-144 — a silent empty index would otherwise
     surface as a confusing FileNotFoundError deep inside run_eval.py).

Must run AFTER scripts/prepare_units_db.py (same --split).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# exp6/ (parent of scripts/) holds config.py and utils.py — not on sys.path
# when this file is launched directly as `python scripts/prep_index.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from utils import logger


def run(cmd: list[str]) -> None:
    logger.info("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def index_files(repo_path: str, project: str) -> None:
    run([config.SIMARGL_CMD, "index", "files", repo_path, "--project", project])


def index_units(train_db_path: str, project: str) -> None:
    run([
        config.SIMARGL_CMD, "index", "units", train_db_path,
        "--project", project, "--mode", "tasks",
    ])


def check_status(project: str) -> None:
    result = subprocess.run(
        [config.SIMARGL_CMD, "status", "--project", project],
        capture_output=True, text=True, check=True,
    )
    out = result.stdout
    files_match = re.search(r"Files:\s+(\d+)", out)
    units_match = re.search(r"Units:\s+(\d+)", out)
    n_files = int(files_match.group(1)) if files_match else 0
    n_units = int(units_match.group(1)) if units_match else 0

    logger.info(f"  status: files={n_files}  units={n_units}")
    if n_files == 0:
        raise RuntimeError(
            f"Project '{project}': files index is empty after `index files` — "
            f"mode=file/aggr will fail. Check repo_path in {config.PROJECTS_MANIFEST}."
        )
    if n_units == 0:
        raise RuntimeError(
            f"Project '{project}': units index is empty after `index units` — "
            f"mode=task/aggr will fail. Check that the train-only DB has TASKS rows "
            f"(run prepare_units_db.py first)."
        )


def main():
    parser = argparse.ArgumentParser(description="Index files + train-only units for every project")
    parser.add_argument('--split', choices=list(config.SPLIT_STRATEGIES), default='modn',
                        help="Must match the --split used in prepare_units_db.py")
    parser.add_argument('--project', default=None, help="Limit to one project name from projects.csv")
    args = parser.parse_args()

    projects = config.load_projects()
    if args.project:
        projects = [p for p in projects if p['name'] == args.project]
        if not projects:
            raise SystemExit(f"Project '{args.project}' not found in {config.PROJECTS_MANIFEST}")

    for proj in projects:
        name, repo_path = proj['name'], proj['repo_path']
        train_db = config.train_db_path(name, args.split)

        logger.info("=" * 80)
        logger.info(f"Project: {name}")
        try:
            logger.info(f"Indexing files from: {repo_path}")
            index_files(repo_path, name)

            logger.info(f"Indexing units from: {train_db}")
            index_units(train_db, name)

            check_status(name)
        except subprocess.CalledProcessError as e:
            logger.error(f"Indexing failed for project '{name}': {e}")
            sys.exit(1)
        except RuntimeError as e:
            logger.error(str(e))
            sys.exit(1)

    logger.info("=" * 80)
    logger.info("All projects indexed successfully.")


if __name__ == "__main__":
    main()
