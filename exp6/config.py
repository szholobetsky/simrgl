# Configuration for exp6 Experiment
# See IMPLEMENTATION_DETAIL_CL.md for the full rationale behind every choice here.

import csv
import os

# Paths (relative to exp6/, matches how scripts/ and run_eval.py are invoked)
PROJECTS_MANIFEST = "data/projects.csv"   # name,repo_path,db_path — filled in on the run machine
RESULTS_DIR = "results/"

TRAIN_DB_PATH_TMPL = "data/db/{project}_{split}_train.db"   # written by scripts/prepare_units_db.py
TEST_SET_TMPL = "data/test_set_{project}.json"               # written by scripts/prepare_units_db.py


def load_projects(manifest_path: str = PROJECTS_MANIFEST) -> list[dict]:
    """Read the project manifest: [{"name":..., "repo_path":..., "db_path":...}, ...].

    No project paths are hardcoded anywhere else in exp6 — everything comes from
    this file, which is filled in on the machine where the experiment actually runs.
    """
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"name", "repo_path", "db_path"}
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{manifest_path}: row {row} missing columns {missing}")
    return rows


def train_db_path(project: str, split: str) -> str:
    return TRAIN_DB_PATH_TMPL.format(project=project, split=split)


def test_set_path(project: str) -> str:
    return TEST_SET_TMPL.format(project=project)


# Split strategy: 'recent' inflates module-level MAP via temporal proximity;
# 'modn' is the honest baseline (see AGENTS.md locked finding #2). Copied
# verbatim from exp3.2/config.py:229-238.
SPLIT_STRATEGIES = {
    'recent': {
        'name': 'Recent Split',
        'description': 'Test on most recent N tasks'
    },
    'modn': {
        'name': 'ModN Split',
        'description': 'Test on every k-th task (uniform sampling)'
    }
}

# Time window over training data, applied before indexing (exp3.2/config.py:210-226).
WINDOW_VARIANTS = {
    'w100': {
        'name': 'NEAREST 100',
        'description': 'Train on last 100 tasks before test',
        'size': 100
    },
    'w1000': {
        'name': 'NEAREST 1000',
        'description': 'Train on last 1000 tasks before test',
        'size': 1000
    },
    'all': {
        'name': 'ALL',
        'description': 'Train on all available history',
        'size': None
    }
}

# Evaluation parameters
TEST_SIZE = 200
K_VALUES = [5, 10, 20]          # P@K / R@K reporting granularity (exp6-specific)
TOP_K_VALUES = [1, 3, 5, 10]    # unified with exp3/exp3.1/exp3.2 for cross-experiment comparison

# RQ3 prior-filtering grid (user decision, IMPLEMENTATION_DETAIL_CL.md section 6/10):
# asymmetric penalties to see sensitivity to filter "shoulder width", not a single value.
LAMBDA_GRID = [0.2, 0.5, 0.7]
TOP_M_MODULES = 5
TOP_M_EXTENSIONS = 5

# Default simargl CLI settings
SIMARGL_CMD = "simargl"
DEFAULT_TOP_N = 20
