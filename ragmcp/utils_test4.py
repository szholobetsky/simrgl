import pytest
from utils import *  # Assuming these are the functions to test

def test_preprocess_text():
    assert preprocess_text(None) == ""
    assert preprocess_text(0.5) == "0.5"
    assert preprocess_text("  Hello  ") == "Hello"
    assert preprocess_text("NaN") == ""

def test_combine_text_fields():
    assert combine_text_fields({"a": "x", "b": "y"}, ["a", "b"]) == "x y"
    assert combine_text_fields({"x": "1"}, ["x"]) == "1"

def test_extract_file_path():
    assert extract_file_path("C:\test") == "test"
    assert extract_file_path("") == "root"
    assert extract_file_path("src/main/Foo.java") == "src"

def test_extract_module_path():
    assert extract_module_path("server/api/handler.py") == "server"
    assert extract_module_path("root") == "root"
    assert extract_module_path("standalone.js") == "root"

def test_calculate_metrics_for_query():
    # Test case with relevant_files and k_values
    metrics = {
        'AP': 0.8,
        'RR': 0.1,
        'P@1': 0.5,
        'P@2': 0.3
    }
    assert calculate_metrics_for_query(retrieved_paths=["path1", "path2"], 
                                      relevant_files=["file1"], k_values=[1,2]) == {
        'AP': 0.4, 'RR': 0.05,
        'P@1': 0.5, 'P@2': 0.3
    }

def test_aggregate_metrics():
    aggregated = aggregate_metrics([
        {'AP': 0.8, 'RR': 0.1},
        {'P@1': 0.5, 'R@1': 0.2}
    ], [1, 2])
    assert aggregated['MAP'] == 0.4
    assert aggregated['MRR'] == 0.05
    assert aggregated['P@1'] == 0.5
    assert aggregated['R@1'] == 0.2

def test_format_metrics_row():
    metrics = {
        'MAP': 0.8,
        'MRR': 0.05,
        'P@1': 0.5,
        'P@2': 0.3
    }
    assert format_metrics_row("exp1", "exp2", metrics, [1, 2]) == {
        'MAP': 0.8,
        'MRR': 0.05,
        'P@1': 0.5,
        'P@2': 0.3
    }

def test_log_experiment_start():
    with pytest.mark.unit():
        log_experiment_start("exp1", {"config": "test_config"})
        assert logger.info_called  # Assuming logger is mocked, but hard to test directly

def test_validate_test_set():
    test_cases = [
        {"test_set": [{"NAME", "TITLE"}, {"relevant_files": ["file1"]}]},
        {"test_set": []},
        {"test_set": {"NAME": "Test", "TITLE": "Test", "relevant_files": ["file1"]}}
    ]
    for test_case in test_cases:
        assert validate_test_set(test_case) == True
        assert validate_test_set(test_case) == False

# Additional tests for edge cases:
def test_extract_file_path_empty():
    assert extract_file_path("") == "root"

def test_extract_file_path_with_slashes():
    assert extract_file_path("C:\\test\\file.txt") == "test\\file.txt"

def test_extract_file_path_with_underscores():
    assert extract_file_path("my_project/path") == "my_project/path"
