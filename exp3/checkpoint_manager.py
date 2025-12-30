"""
Checkpoint Manager for Resumable Experiments
Tracks completed experiment variants to enable resume after interruption
"""

import json
import os
from typing import Dict, List, Set
from datetime import datetime
from utils import logger


class CheckpointManager:
    """Manages experiment checkpoints for resume capability"""

    def __init__(self, checkpoint_file: str = "experiment_results/checkpoint.json"):
        self.checkpoint_file = checkpoint_file
        self.checkpoint_dir = os.path.dirname(checkpoint_file)
        self.data = self._load_checkpoint()

    def _ensure_dir(self):
        """Create checkpoint directory if it doesn't exist"""
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            logger.info(f"Created checkpoint directory: {self.checkpoint_dir}")

    def _load_checkpoint(self) -> Dict:
        """Load checkpoint from file"""
        self._ensure_dir()

        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded checkpoint from {self.checkpoint_file}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
                return self._create_new_checkpoint()
        else:
            return self._create_new_checkpoint()

    def _create_new_checkpoint(self) -> Dict:
        """Create new checkpoint structure"""
        return {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "completed_etl": [],
            "completed_experiments": [],
            "failed_etl": [],
            "failed_experiments": [],
            "current_model": None,
            "current_strategy": None
        }

    def _save_checkpoint(self):
        """Save checkpoint to file"""
        self._ensure_dir()

        self.data["last_updated"] = datetime.now().isoformat()

        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.data, f, indent=2)
            logger.debug(f"Checkpoint saved to {self.checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def mark_etl_completed(self, model: str, strategy: str, source: str, target: str, window: str):
        """Mark an ETL variant as completed"""
        variant_id = f"{model}_{strategy}_{source}_{target}_{window}"

        if variant_id not in self.data["completed_etl"]:
            self.data["completed_etl"].append(variant_id)
            self._save_checkpoint()
            logger.info(f"✓ ETL completed: {variant_id}")

    def mark_etl_failed(self, model: str, strategy: str, source: str, target: str, window: str, error: str):
        """Mark an ETL variant as failed"""
        variant_id = f"{model}_{strategy}_{source}_{target}_{window}"

        failure_entry = {
            "variant_id": variant_id,
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        }

        self.data["failed_etl"].append(failure_entry)
        self._save_checkpoint()
        logger.error(f"✗ ETL failed: {variant_id} - {error}")

    def mark_experiment_completed(self, model: str, strategy: str, source: str, target: str, window: str):
        """Mark an experiment variant as completed"""
        variant_id = f"{model}_{strategy}_{source}_{target}_{window}"

        if variant_id not in self.data["completed_experiments"]:
            self.data["completed_experiments"].append(variant_id)
            self._save_checkpoint()
            logger.info(f"✓ Experiment completed: {variant_id}")

    def mark_experiment_failed(self, model: str, strategy: str, source: str, target: str, window: str, error: str):
        """Mark an experiment variant as failed"""
        variant_id = f"{model}_{strategy}_{source}_{target}_{window}"

        failure_entry = {
            "variant_id": variant_id,
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        }

        self.data["failed_experiments"].append(failure_entry)
        self._save_checkpoint()
        logger.error(f"✗ Experiment failed: {variant_id} - {error}")

    def is_etl_completed(self, model: str, strategy: str, source: str, target: str, window: str) -> bool:
        """Check if ETL variant is already completed"""
        variant_id = f"{model}_{strategy}_{source}_{target}_{window}"
        return variant_id in self.data["completed_etl"]

    def is_experiment_completed(self, model: str, strategy: str, source: str, target: str, window: str) -> bool:
        """Check if experiment variant is already completed"""
        variant_id = f"{model}_{strategy}_{source}_{target}_{window}"
        return variant_id in self.data["completed_experiments"]

    def set_current_progress(self, model: str, strategy: str):
        """Update current progress"""
        self.data["current_model"] = model
        self.data["current_strategy"] = strategy
        self._save_checkpoint()

    def get_completed_count(self) -> Dict[str, int]:
        """Get count of completed variants"""
        return {
            "etl": len(self.data["completed_etl"]),
            "experiments": len(self.data["completed_experiments"]),
            "failed_etl": len(self.data["failed_etl"]),
            "failed_experiments": len(self.data["failed_experiments"])
        }

    def clear_checkpoint(self):
        """Clear checkpoint (start fresh)"""
        logger.info("Clearing checkpoint - starting fresh")
        self.data = self._create_new_checkpoint()
        self._save_checkpoint()

    def get_summary(self) -> str:
        """Get human-readable summary"""
        counts = self.get_completed_count()

        summary = [
            "=" * 70,
            "CHECKPOINT STATUS",
            "=" * 70,
            f"Created: {self.data.get('created_at', 'Unknown')}",
            f"Last Updated: {self.data.get('last_updated', 'Unknown')}",
            "",
            f"Completed ETL: {counts['etl']}",
            f"Completed Experiments: {counts['experiments']}",
            f"Failed ETL: {counts['failed_etl']}",
            f"Failed Experiments: {counts['failed_experiments']}",
            ""
        ]

        if self.data.get("current_model"):
            summary.append(f"Current Model: {self.data['current_model']}")
            summary.append(f"Current Strategy: {self.data['current_strategy']}")

        summary.append("=" * 70)
        return "\n".join(summary)

    def should_resume(self) -> bool:
        """Check if there's an existing checkpoint to resume from"""
        counts = self.get_completed_count()
        return counts['etl'] > 0 or counts['experiments'] > 0
