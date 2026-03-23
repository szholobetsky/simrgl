# utils_test.py
"""
Unit‑tests for utils.py

Run with:
    pytest -q utils_test.py
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ----------------------------------------------------------------------
# Import the functions we want to test.
# ----------------------------------------------------------------------
from utils import (
    preprocess_text,
    combine_text_fields,
    extract_file_path,
    extract_module_path,
    calculate_metrics_for_query,
    aggregate_metrics,
    format_metrics_row,
    log_experiment_start,
    log_experiment_complete,
    validate_test_set,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def dummy_task():
    """
    Return a minimal task dict that satisfies the validation logic.
    The fields are deliberately empty – the only thing we care about is the shape.
    """
    return {"NAME": "Dummy", "TITLE": "Dummy", "relevant_files": ["a.txt"]}


@pytest.fixture
def test_set():
    """A small, valid test‑set with three tasks."""
    return [
        {
            "NAME": "Task 1",
            "TITLE": "Task 1 title",
            "relevant_files": ["src/main/file1.py", "/tmp/good.txt"],
        },
        {
            "NAME": "Task 2",
            "TITLE": "Task 2 title",
            "relevant_files": ["data.csv", "src/utils/helpers.py"],
        },
        {
            "NAME": "Task 3",
            "TITLE": "Task 3 title",
            # missing a file – will be reported as error
            "relevant_files": [],
        },
    ]


# ----------------------------------------------------------------------
# Helper data used by several tests
# ----------------------------------------------------------------------
RETRIEVED = ["/src/fileA.txt", "/tmp/bad.txt", "/src/fileB.txt"]
RELEVANT = {"/src/fileA.txt", "/src/fileB.txt"}
K_VALUES = [1, 3]


def test_preprocess_text():
    assert preprocess_text(None) == ""
    assert preprocess_text("   \tfoo\n") == "foo"
    # NaN is converted to empty string
    nan_val = np.nan
    assert preprocess_text(nan_val) == ""


def test_combine_text_fields(dummy_task, dummy_task2):
    """Combine two rows with overlapping fields."""
    row1 = {"NAME": "A", "TITLE": "Title 1"}
    row2 = {"NAME": "B", "TITLE": "Title 2"}
    combined = combine_text_fields(row1, ["NAME", "TITLE"])
    assert combined == "A Title 1 B Title 2"

    # Empty fields become “empty_task”
    empty_row = {}
    combined_empty = combine_text_fields(empty_row, [])
    assert combined_empty == "empty_task"


def test_extract_file_path(tmp_path):
    """Normal path → normalized POSIX style."""
    p = tmp_path / "src\\main\\Foo.java"
    assert extract_file_path(str(p)) == str(p).replace("\\", "/")

    # Empty string → unknown
    assert extract_file_path("") == "unknown"

    # Already POSIX
    assert extract_file_path("/tmp/file.txt") == "/tmp/file.txt"


def test_extract_module_path(tmp_path):
    """Extract the first directory component (or “root”)."""
    p = tmp_path / "src\\main\\Foo.java"
    assert extract_module_path(str(p)) == "src"

    # Single file with no sub‑dirs
    single = str(tmp_path / "foo.js")
    assert extract_module_path(single) == "root"


def test_calculate_metrics_for_query():
    """Happy path – non‑empty relevant set."""
    metrics = calculate_metrics_for_query(RETRIEVED, RELEVANT, K_VALUES)
    # AP: 2/3 ≈ 0.6667
    assert pytest.approx(metrics["AP"], abs=1e-4) == 2 / 3

    # RR: first hit at rank 1 → 1.0
    assert metrics["RR"] == 1.0

    # P@1 = 1/1, R@1 = 1/2 (only one relevant in top‑k)
    assert pytest.approx(metrics["P@1"], abs=1e-4) == 1.0
    assert pytest.approx(metrics["R@1"], abs=1e-4) == 0.5

    # P@3 = 2/3, R@3 = 2/2 = 1.0
    assert pytest.approx(metrics["P@3"], abs=1e-4) == 2 / 3
    assert metrics["R@3"] == 1.0


def test_calculate_metrics_for_query_empty_relevant():
    """All metrics should be zero when no relevant files exist."""
    metrics = calculate_metrics_for_query(RETRIEVED, set(), K_VALUES)
    for k in K_VALUES:
        assert metrics[f"P@{k}"] == 0.0
        assert metrics[f"R@{k}"] == 0.0
    assert metrics["AP"] == 0.0
    assert metrics["RR"] == 0.0


def test_calculate_metrics_for_query_no_retrieved():
    """Even if relevant set non‑empty, nothing in retrieved list → zero."""
    metrics = calculate_metrics_for_query([], RELEVANT, K_VALUES)
    for k in K_VALUES:
        assert metrics[f"P@{k}"] == 0.0
        assert metrics[f"R@{k}"] == 0.0


def test_aggregate_metrics():
    """Average across two identical queries."""
    m1 = calculate_metrics_for_query(RETRIEVED, RELEVANT, K_VALUES)
    m2 = calculate_metrics_for_query(RETRIEVED, RELEVANT, K_VALUES)
    agg = aggregate_metrics([m1, m2], K_VALUES)

    assert agg["MAP"] == pytest.approx(m1["AP"])
    assert agg["MRR"] == pytest.approx(m1["RR"])
    for k in K_VALUES:
        assert agg[f"P@{k}"] == pytest.approx(metrics = m1[f"P@{k}"])
        assert agg[f"R@{k}"] == pytest.approx(metrics = m1[f"R@{k}"])


def test_aggregate_metrics_empty():
    """Empty list → empty dict (no division by zero)."""
    agg = aggregate_metrics([], K_VALUES)
    assert agg == {}


def test_format_metrics_row(tmp_path):
    """Create a row that could be written to CSV."""
    metrics = calculate_metrics_for_query(RETRIEVED, RELEVANT, K_VALUES)

    row = format_metrics_row(
        experiment_id="exp_01",
        source="simple",
        target="txt",
        window="small",
        split="train",
        metrics=metrics,
        k_values=K_VALUES,
    )
    # Expected values (same as m1 from above test)
    assert row["MAP"] == pytest.approx(2 / 3, abs=1e-4)
    assert row["MRR"] == 1.0
    assert row["P@1"] == 1.0
    assert row["R@1"] == 0.5
    assert row["P@3"] == pytest.approx(2 / 3, abs=1e-4)
    assert row["R@3"] == 1.0

    # The experiment_id path is unique – never a duplicate key.
    assert len(set(row.keys())) == len(row)


def test_log_experiment_start(tmp_path):
    """Log should create two handlers and write lines to file & console."""
    config = {"tmp": tmp_path}
    log_experiment_start("exp_02", config)

    # The logger writes INFO to both stream + file.
    assert "Starting Experiment: exp_02" in os.popen("cat experiment.log").read()
    assert "Configuration: {" in os.popen("cat experiment.log").read()


def test_log_experiment_complete(tmp_path):
    """Log completion with a float metric."""
    metrics = calculate_metrics_for_query(RETRIEVED, RELEVANT, K_VALUES)
    log_experiment_complete("exp_03", metrics)

    assert "Completed Experiment: exp_03" in os.popen("cat experiment.log").read()
    assert f"MAP={metrics['MAP']:.4f}" in os.popen("cat experiment.log").read()


def test_validate_test_set():
    """Valid set passes; missing fields fails."""
    valid = [
        {"NAME": "A", "TITLE": "B", "relevant_files": ["x.txt"]},
    ]
    assert validate_test_set(valid) is True
    assert validate_test_set([]) is False  # empty list

    bad = [{"NAME": "C", "TITLE": "D"}]   # missing required field
    with pytest.raises(logging.LoggerError):
        _ = validate_test_set(bad)


# ----------------------------------------------------------------------
# Test that the logging module is actually used (no double‑printing)
# ----------------------------------------------------------------------
def test_logging_is_monotonic():
    """Two consecutive calls produce a monotonic log file."""
    logger.info("First")
    logger.info("Second")

    content = os.popen("cat experiment.log").read()
    lines = [ln for ln in content.splitlines() if "First" not in ln and "Second" not in ln]
    assert all(l > prev for l, prev in zip(lines[1:], lines[:-1]))


# ----------------------------------------------------------------------
# Run‑time sanity check – make sure the module imports cleanly
# ----------------------------------------------------------------------
def test_imports_ok():
    """Basic smoke test that utils can be imported."""
    from utils import calculate_metrics_for_query, aggregate_metrics
    # Nothing to assert here; just confirms no runtime error.
    pass
