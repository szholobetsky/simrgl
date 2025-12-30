"""
GPU Memory Management Utilities
Handles GPU memory cleanup and adaptive batch size/device selection
"""

import gc
import torch
from typing import Optional, Tuple, Dict
from utils import logger


def clear_gpu_memory():
    """Clear GPU memory cache and run garbage collection"""
    try:
        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU memory cleared")
    except Exception as e:
        logger.warning(f"Failed to clear GPU memory: {e}")


def get_gpu_memory_info() -> Optional[Dict]:
    """Get GPU memory information"""
    try:
        if not torch.cuda.is_available():
            return None

        gpu_id = 0  # Assuming single GPU
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        allocated_memory = torch.cuda.memory_allocated(gpu_id)
        reserved_memory = torch.cuda.memory_reserved(gpu_id)
        free_memory = total_memory - allocated_memory

        return {
            "total_gb": total_memory / (1024**3),
            "allocated_gb": allocated_memory / (1024**3),
            "reserved_gb": reserved_memory / (1024**3),
            "free_gb": free_memory / (1024**3)
        }
    except Exception as e:
        logger.warning(f"Failed to get GPU memory info: {e}")
        return None


def log_gpu_memory():
    """Log current GPU memory usage"""
    info = get_gpu_memory_info()
    if info:
        logger.info(
            f"GPU Memory: {info['allocated_gb']:.2f}GB / {info['total_gb']:.2f}GB "
            f"(Free: {info['free_gb']:.2f}GB)"
        )
    else:
        logger.info("GPU not available or unable to get memory info")


def should_use_cpu(model_name: str, available_memory_gb: float = None) -> bool:
    """
    Determine if model should use CPU based on size and available memory

    Args:
        model_name: Name of the model
        available_memory_gb: Available GPU memory in GB (None = auto-detect)

    Returns:
        True if should use CPU, False if can use GPU
    """
    # If no GPU available, always use CPU
    if not torch.cuda.is_available():
        logger.info("No GPU available - using CPU")
        return True

    # Get available memory
    if available_memory_gb is None:
        info = get_gpu_memory_info()
        if info:
            available_memory_gb = info['free_gb']
        else:
            # If can't determine, be conservative
            logger.warning("Cannot determine GPU memory - defaulting to CPU")
            return True

    # Estimate model memory requirements (rough estimates)
    # These are approximate - actual usage depends on batch size and sequence length
    model_memory_estimates = {
        'bge-small': 0.5,      # ~384 dim, small model
        'bge-large': 1.5,      # ~1024 dim, larger model
        'bge-m3': 2.0,         # ~1024 dim, multilingual, larger
        'gte-qwen2': 3.0,      # ~1536 dim, 1.5B params
        'nomic-embed': 1.2,    # ~768 dim
        'gte-large': 1.5,      # ~1024 dim
        'e5-large': 1.5        # ~1024 dim
    }

    # Find model key in name
    required_memory = 2.0  # Default conservative estimate
    for key, memory in model_memory_estimates.items():
        if key in model_name.lower():
            required_memory = memory
            break

    # Add buffer for batch processing (multiply by 2 for safety)
    required_memory *= 2

    # Decision
    use_cpu = available_memory_gb < required_memory

    if use_cpu:
        logger.warning(
            f"Insufficient GPU memory for {model_name}: "
            f"Required ~{required_memory:.1f}GB, Available {available_memory_gb:.1f}GB - Using CPU"
        )
    else:
        logger.info(
            f"Sufficient GPU memory for {model_name}: "
            f"Required ~{required_memory:.1f}GB, Available {available_memory_gb:.1f}GB - Using GPU"
        )

    return use_cpu


def get_adaptive_batch_size(model_name: str, default_batch_size: int = 32) -> int:
    """
    Get adaptive batch size based on model and available memory

    Args:
        model_name: Name of the model
        default_batch_size: Default batch size

    Returns:
        Recommended batch size
    """
    # Large models need smaller batch sizes
    large_models = ['bge-m3', 'gte-qwen2', 'qwen', '1.5b', '1b']

    for keyword in large_models:
        if keyword in model_name.lower():
            batch_size = min(default_batch_size, 8)
            logger.info(f"Using smaller batch size {batch_size} for large model {model_name}")
            return batch_size

    return default_batch_size


def cleanup_model(model):
    """
    Properly cleanup a model to free memory

    Args:
        model: The model to cleanup
    """
    try:
        # Move model to CPU if it's on GPU
        if hasattr(model, 'to'):
            model.to('cpu')

        # Delete model
        del model

        # Clear memory
        clear_gpu_memory()

        logger.info("Model cleaned up successfully")
    except Exception as e:
        logger.warning(f"Error during model cleanup: {e}")
