"""
Utility functions for RAG Research Experiment
Includes metrics calculation, text preprocessing, and path extraction
"""

import numpy as np
import logging
from typing import List, Set, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def preprocess_text(text: Any) -> str:
    """Clean and normalize text for embedding"""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    return str(text).strip()


def combine_text_fields(row: Dict[str, Any], fields: List[str]) -> str:
    """Combine multiple text fields into a single string"""
    texts = [preprocess_text(row.get(field, "")) for field in fields]
    combined = " ".join(filter(None, texts))
    return combined if combined else "empty_task"


def extract_file_path(path: str) -> str:
    """Extract normalized file path"""
    if not path:
        return "unknown"
    return str(path).replace('\\', '/').strip()


def extract_module_path(path: str) -> str:
    """
    Extract module path (root folder) from file path
    Examples:
        'src/main/Foo.java' -> 'src'
        'server/api/handler.py' -> 'server'
        'standalone.js' -> 'root'
    """
    if not path:
        return "root"

    normalized_path = str(path).replace('\\', '/').strip()
    parts = normalized_path.split('/')

    if len(parts) > 1:
        return parts[0]
    return "root"


def calculate_metrics_for_query(
    retrieved_paths: List[str],
    relevant_files: Set[str],
    k_values: List[int]
) -> Dict[str, float]:
    """
    Calculate retrieval metrics for a single query

    Args:
        retrieved_paths: List of retrieved file paths (in rank order)
        relevant_files: Set of ground truth relevant files
        k_values: List of k values for P@K and R@K

    Returns:
        Dictionary with metric values
    """
    metrics = {}

    if not relevant_files:
        # No ground truth files - all metrics are 0
        metrics['AP'] = 0.0
        metrics['RR'] = 0.0
        for k in k_values:
            metrics[f'P@{k}'] = 0.0
            metrics[f'R@{k}'] = 0.0
        return metrics

    # Calculate hit positions and precision at each hit
    hits = 0
    precisions_at_hits = []
    first_hit_rank = 0

    for rank, path in enumerate(retrieved_paths, 1):
        if path in relevant_files:
            hits += 1
            precisions_at_hits.append(hits / rank)
            if first_hit_rank == 0:
                first_hit_rank = rank

    # Average Precision (AP)
    if precisions_at_hits:
        metrics['AP'] = sum(precisions_at_hits) / len(relevant_files)
    else:
        metrics['AP'] = 0.0

    # Reciprocal Rank (RR)
    metrics['RR'] = 1.0 / first_hit_rank if first_hit_rank > 0 else 0.0

    # Precision@K and Recall@K for each k
    for k in k_values:
        top_k_paths = retrieved_paths[:k]
        hits_at_k = sum(1 for path in top_k_paths if path in relevant_files)

        metrics[f'P@{k}'] = hits_at_k / k if k > 0 else 0.0
        metrics[f'R@{k}'] = hits_at_k / len(relevant_files)

    return metrics


def aggregate_metrics(
    all_metrics: List[Dict[str, float]],
    k_values: List[int]
) -> Dict[str, float]:
    """
    Aggregate metrics across all queries

    Args:
        all_metrics: List of metric dictionaries (one per query)
        k_values: List of k values used

    Returns:
        Dictionary with aggregated metrics (means)
    """
    if not all_metrics:
        return {}

    aggregated = {}

    # Mean Average Precision (MAP)
    aggregated['MAP'] = np.mean([m['AP'] for m in all_metrics])

    # Mean Reciprocal Rank (MRR)
    aggregated['MRR'] = np.mean([m['RR'] for m in all_metrics])

    # Mean P@K and R@K
    for k in k_values:
        aggregated[f'P@{k}'] = np.mean([m[f'P@{k}'] for m in all_metrics])
        aggregated[f'R@{k}'] = np.mean([m[f'R@{k}'] for m in all_metrics])

    return aggregated


def format_metrics_row(
    experiment_id: str,
    source: str,
    target: str,
    window: str,
    split: str,
    metrics: Dict[str, float],
    k_values: List[int]
) -> Dict[str, Any]:
    """
    Format metrics into a row for CSV output

    Args:
        experiment_id: Unique experiment identifier
        source: Source variant name
        target: Target variant name
        window: Window variant name
        split: Split strategy name
        metrics: Aggregated metrics dictionary
        k_values: List of k values used

    Returns:
        Dictionary representing a row in the results table
    """
    row = {
        'experiment_id': experiment_id,
        'source': source,
        'target': target,
        'window': window,
        'split': split,
        'MAP': metrics['MAP'],
        'MRR': metrics['MRR']
    }

    for k in k_values:
        row[f'P@{k}'] = metrics[f'P@{k}']
        row[f'R@{k}'] = metrics[f'R@{k}']

    return row


def log_experiment_start(experiment_id: str, config: Dict[str, Any]):
    """Log the start of an experiment with configuration"""
    logger.info(f"=" * 80)
    logger.info(f"Starting Experiment: {experiment_id}")
    logger.info(f"Configuration: {config}")
    logger.info(f"=" * 80)


def log_experiment_complete(experiment_id: str, metrics: Dict[str, float]):
    """Log the completion of an experiment with results"""
    logger.info(f"Completed Experiment: {experiment_id}")
    logger.info(f"Results: MAP={metrics['MAP']:.4f}, MRR={metrics['MRR']:.4f}")
    logger.info(f"-" * 80)


def validate_test_set(test_set: List[Dict[str, Any]]) -> bool:
    """Validate test set structure"""
    if not test_set:
        logger.error("Test set is empty")
        return False

    required_fields = ['NAME', 'TITLE', 'relevant_files']
    for i, task in enumerate(test_set):
        for field in required_fields:
            if field not in task:
                logger.error(f"Test set task {i} missing required field: {field}")
                return False

    logger.info(f"Test set validation passed: {len(test_set)} tasks")
    return True
