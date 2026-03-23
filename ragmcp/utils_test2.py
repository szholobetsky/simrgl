import pytest
from utils import preprocess_text, combine_text_fields, extract_file_path, extract_module_path, calculate_metrics_for_query, aggregate_metrics, format_metrics_row, log_experiment_start, log_experiment_complete, validate_test_set

def test_preprocess_text():
    assert preprocess_text(None) == ""
    assert preprocess_text("Hello") == "Hello"
    assert preprocess_text("NaN") == ""
    assert preprocess_text(3.14) == "3.14"
    assert preprocess_text("   ") == ""

def test_combine_text_fields():
    assert combine_text_fields({"a": "value1", "b": "value2"}, ["a", "b"]) == "value1 value2"
    assert combine_text_fields({"a": "value1"}, ["a"]) == "value1"
    assert combine_text_fields({"x": "test"}, ["x", "y"]) == "test"

def test_extract_file_path():
    assert extract_file_path("C:/test.txt") == "C:/test"
    assert extract_file_path("/path/to/file") == "/path/to/file"
    assert extract_file_path("file\\path") == "file/path"

def test_extract_module_path():
    assert extract_module_path("src/main/file") == "src"
    assert extract_module_path("server/api/config") == "server"
    assert extract_module_path("root") == "root"

def test_calculate_metrics_for_query():
    metrics = {
        "AP": 0.8,
        "RR": 0.95,
        "P@1": 0.1,
        "P@2": 0.05,
        "R@1": 0.2
    }
    assert calculate_metrics_for_query(["file1.txt"], {"relevant_files": ["file1.txt"], "k_values": [0, 1]}) == {
        "AP": 0.1,
        "RR": 0.95,
        "P@1": 0.1,
        "P@2": 0.05,
        "R@1": 0.2,
        "AP": 0.05,
        "RR": 0.95
    }

def test_aggregate_metrics():
    metrics = {
        "AP": 0.85,
        "RR": 0.9,
        "P@1": 0.3,
        "P@2": 0.2
    }
    assert aggregate_metrics(metrics, [0, 1]) == {
        "MAP": 0.85,
        "MRR": 0.9,
        "P@1": 0.3,
        "P@2": 0.2
    }

def test_format_metrics_row():
    row = {
        "experiment_id": "exp123",
        "source": "A",
        "target": "B",
        "window": "temp",
        "split": "all",
        "MAP": 0.95,
        "MRR": 0.95,
        "P@1": 0.5,
        "P@2": 0.3
    }
    assert format_metrics_row(experiment_id="exp123", source="A", target="B", window="temp", split="all", metrics={"MAP":0.95, "MRR":0.95}, k_values=[1,2]) == {
        "experiment_id": "exp123",
        "source": "A",
        "target": "B",
        "window": "temp",
        "split": "all",
        "MAP": 0.95,
        "MRR": 0.95,
        "P@1": 0.5,
        "P@2": 0.3,
        "P@1": 0.5,
        "P@2": 0.3
    }

def test_validate_test_set():
    test_cases = [
        {"task": "Test case 1", "required_fields": ["NAME", "TITLE"]},
        {"task": "Test case 2", "required_fields": ["relevant_files"]},
        {"task": "Test case 3", "k_values": [0, 1]}
    ]
    for case in test_cases:
        assert validate_test_set(case) == True
