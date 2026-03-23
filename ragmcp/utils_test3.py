import unittest
from unittest.mock import patch, MagicMock
import numpy as np
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
    validate_test_set
)

class TestUtils(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.test_texts = [
            "This is a sample text.",
            "Another example text with different casing.",
            None,
            42.0,
            np.nan,
            ""
        ]
        self.fields = ["text", "float"]
        self.k_values = [2, 3]
        self.test_paths = [
            "/docs/file1.txt",
            "src/main/Foo.java",
            "server/api/handler.py",
            "root/another_file.txt"
        ]
        self.relevant_files = {"file1.txt", "src/main/Foo.java"}
        self.retrieved_paths = ["/docs/file1.txt", "src/main/Foo.java"]

    def test_preprocess_text(self):
        self.assertEqual(preprocess_text(""), "")
        self.assertEqual(preprocess_text("  text  "), "text")
        self.assertEqual(preprocess_text(42.0), "42.0")
        self.assertEqual(preprocess_text(np.nan), "")
        self.assertEqual(preprocess_text(""), "")
        self.assertEqual(preprocess_text(""), "")

    def test_combine_text_fields(self):
        result = combine_text_fields({"text": "test"}, self.fields)
        self.assertEqual(result, "test")

        result = combine_text_fields({"text": "test"}, ["text"])
        self.assertEqual(result, "test")

    def test_extract_file_path(self):
        self.assertEqual(extract_file_path("/docs/file1.txt"), "/docs")
        self.assertEqual(extract_file_path(""), "unknown")
        self.assertEqual(extract_file_path("\\folder\\file.txt"), "root")
        self.assertEqual(extract_file_path("/absolute/path"), "/absolute/path")

    def test_extract_module_path(self):
        self.assertEqual(extract_module_path("/docs"), "docs")
        self.assertEqual(extract_module_path("src/main"), "src")
        self.assertEqual(extract_module_path("server"), "server")
        self.assertEqual(extract_module_path("root"), "root")

    def test_calculate_metrics_for_query(self):
        metrics = calculate_metrics_for_query(
            self.test_paths,
            self.relevant_files,
            self.k_values
        )
        self.assertIn('AP', metrics)
        self.assertIn('RR', metrics)
        self.assertEqual(metrics['AP'], 0.0)  # No hits, so AP=0
        self.assertEqual(metrics['RR'], 0.0)
        self.assertEqual(metrics['P2'], 1.0 / 1)
        self.assertEqual(metrics['R2'], 1.0 / 1)

    def test_aggregate_metrics(self):
        metrics = aggregate_metrics(
            [{'AP': 0.5, 'RR': 0.8}, {'AP': 0.3, 'RR': 0.9}],
            self.k_values
        )
        self.assertAlmostEqual(metrics['MAP'], 0.55)
        self.assertAlmostEqual(metrics['MRR'], 0.85)
        self.assertAlmostEqual(metrics['P2'], 0.5)
        self.assertAlmostEqual(metrics['R2'], 0.8)

    def test_format_metrics_row(self):
        metrics = {
            'MAP': 0.55,
            'MRR': 0.85,
            'P2': 0.5,
            'R2': 0.8
        }
        row = format_metrics_row(
            experiment_id="exp1",
            source="test",
            target="target",
            window="test",
            split="split",
            metrics=metrics,
            k_values=[2, 3]
        )
        expected = {
            'experiment_id': 'exp1',
            'source': 'test',
            'target': 'target',
            'window': 'test',
            'split': 'split',
            'MAP': 0.55,
            'MRR': 0.85,
            'P2': 0.5,
            'R2': 0.8
        }
        self.assertEqual(row, expected)

    @patch('logging.Logger')
    def test_log_experiment_start(self, logger):
        log_experiment_start("exp1", {"key": "value"})
        logger.info.assert_called_once_with("=" * 80)
        logger.info.assert_called_with("Starting Experiment: exp1")
        logger.info.assert_called_with("Configuration: {'key': 'value'}")

    @patch('logging.Logger')
    def test_log_experiment_complete(self, logger):
        metrics = {
            'MAP': 0.55,
            'MRR': 0.85
        }
        log_experiment_complete("exp1", metrics)
        logger.info.assert_called_once_with("Completed Experiment: exp1")
        logger.info.assert_called_with("Results: MAP=0.55, MRR=0.85")

    def test_validate_test_set(self):
        test_set = [
            {"NAME": "test1", "TITLE": "Test", "relevant_files": ["file1.txt"]},
            {"NAME": "test2", "TITLE": "Test2", "relevant_files": []}
        ]
        self.assertTrue(validate_test_set(test_set))

        test_set = []
        self.assertFalse(validate_test_set(test_set))

if __name__ == '__main__':
    unittest.main()
