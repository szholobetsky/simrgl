#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task Selector Utility
Handles train/test split for tasks based on various strategies
"""

import sqlite3
import random
from typing import List, Tuple, Set
import properties as props

# Set random seed for reproducible results
random.seed(42)

class TaskSelector:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()
    
    def log(self, message: str):
        """Print log message if verbose mode is enabled"""
        if props.VERBOSE:
            print(f"[TASK_SELECTOR] {message}")
    
    def get_all_task_ids(self) -> List[int]:
        """Get all task IDs that have commit data"""
        cursor = self.conn.execute("""
        SELECT DISTINCT t.ID
        FROM TASK t
        INNER JOIN RAWDATA r ON r.TASK_NAME = t.NAME
        WHERE t.NAME IS NOT NULL
        ORDER BY t.ID
        """)
        
        return [row[0] for row in cursor.fetchall()]
    
    def select_test_tasks_by_strategy(self, all_task_ids: List[int], target_count: int, strategy: str) -> Set[int]:
        """Select test tasks based on the specified strategy"""
        if target_count <= 0:
            return set()
        
        if len(all_task_ids) <= target_count:
            self.log(f"Warning: Requested {target_count} test tasks but only {len(all_task_ids)} tasks available")
            return set(all_task_ids)
        
        self.log(f"Selecting {target_count} test tasks using strategy: {strategy}")
        
        if strategy == 'lastN':
            # Select N tasks with highest IDs
            test_tasks = set(all_task_ids[-target_count:])
            
        elif strategy == 'firstN':
            # Select N tasks with lowest IDs
            test_tasks = set(all_task_ids[:target_count])
            
        elif strategy == 'middleN':
            # Select N tasks from the middle of the range
            start_idx = (len(all_task_ids) - target_count) // 2
            end_idx = start_idx + target_count
            test_tasks = set(all_task_ids[start_idx:end_idx])
            
        elif strategy == 'modN':
            # Select tasks where ID % divisor â‰ˆ 0
            # Calculate divisor to get approximately target_count tasks
            divisor = max(1, len(all_task_ids) // target_count)
            
            test_tasks = set()
            for task_id in all_task_ids:
                if task_id % divisor == 0:
                    test_tasks.add(task_id)
            
            # If we got too many or too few, adjust
            test_tasks_list = list(test_tasks)
            if len(test_tasks_list) > target_count:
                # Randomly sample to get exactly target_count
                test_tasks = set(random.sample(test_tasks_list, target_count))
            elif len(test_tasks_list) < target_count * 0.8:  # If we got less than 80% of target
                # Add some more tasks randomly
                remaining_tasks = [tid for tid in all_task_ids if tid not in test_tasks]
                additional_needed = min(target_count - len(test_tasks), len(remaining_tasks))
                additional_tasks = random.sample(remaining_tasks, additional_needed)
                test_tasks.update(additional_tasks)
        
        else:
            raise ValueError(f"Unknown test task selection strategy: {strategy}")
        
        self.log(f"Selected {len(test_tasks)} test tasks using {strategy} strategy")
        
        # Log the ID ranges for verification
        test_task_list = sorted(list(test_tasks))
        if test_task_list:
            self.log(f"Test task ID range: {min(test_task_list)} - {max(test_task_list)}")
        
        return test_tasks
    
    def get_train_test_split(self) -> Tuple[Set[int], Set[int]]:
        """
        Get train/test split based on configuration
        
        Returns:
            Tuple of (training_task_ids, test_task_ids)
        """
        all_task_ids = self.get_all_task_ids()
        
        if not props.EXCLUDE_TEST_TASKS_FROM_MODEL:
            # No split - all tasks are used for both training and testing
            self.log("Train/test split disabled - using all tasks for both training and testing")
            return set(all_task_ids), set(all_task_ids)
        
        # Select test tasks
        test_tasks = self.select_test_tasks_by_strategy(
            all_task_ids, 
            props.SUMMARY_TEST_TASK_COUNT, 
            props.EXCLUDE_TEST_TASKS_STRATEGY
        )
        
        # Training tasks are all tasks except test tasks
        train_tasks = set(all_task_ids) - test_tasks
        
        self.log(f"Train/test split completed:")
        self.log(f"  Training tasks: {len(train_tasks)}")
        self.log(f"  Test tasks: {len(test_tasks)}")
        self.log(f"  Total tasks: {len(all_task_ids)}")
        
        if len(train_tasks) == 0:
            raise ValueError("No training tasks available after split")
        
        if len(test_tasks) == 0:
            raise ValueError("No test tasks available after split")
        
        return train_tasks, test_tasks
    
    def create_train_test_filter_clause(self, table_alias: str = "t") -> Tuple[str, str]:
        """
        Create SQL WHERE clauses for filtering training and test tasks
        
        Args:
            table_alias: Alias for the TASK table in the query
            
        Returns:
            Tuple of (train_filter_clause, test_filter_clause)
        """
        if not props.EXCLUDE_TEST_TASKS_FROM_MODEL:
            # No filtering needed
            return "", ""
        
        train_tasks, test_tasks = self.get_train_test_split()
        
        # Create SQL IN clauses
        train_ids_str = ",".join(map(str, train_tasks))
        test_ids_str = ",".join(map(str, test_tasks))
        
        train_filter = f"AND {table_alias}.ID IN ({train_ids_str})" if train_ids_str else "AND 1=0"
        test_filter = f"AND {table_alias}.ID IN ({test_ids_str})" if test_ids_str else "AND 1=0"
        
        return train_filter, test_filter
    
    def is_test_task(self, task_id: int) -> bool:
        """Check if a task is in the test set"""
        if not props.EXCLUDE_TEST_TASKS_FROM_MODEL:
            return True  # All tasks are test tasks when split is disabled
        
        _, test_tasks = self.get_train_test_split()
        return task_id in test_tasks
    
    def print_split_summary(self):
        """Print summary of the train/test split"""
        print("\n" + "="*60)
        print("TRAIN/TEST SPLIT SUMMARY")
        print("="*60)
        
        if not props.EXCLUDE_TEST_TASKS_FROM_MODEL:
            print("Train/test split: DISABLED")
            print("All tasks are used for both training and testing")
            
            all_task_ids = self.get_all_task_ids()
            print(f"Total tasks: {len(all_task_ids)}")
        else:
            print("Train/test split: ENABLED")
            print(f"Strategy: {props.EXCLUDE_TEST_TASKS_STRATEGY}")
            print(f"Target test tasks: {props.SUMMARY_TEST_TASK_COUNT}")
            
            train_tasks, test_tasks = self.get_train_test_split()
            
            print(f"Training tasks: {len(train_tasks)}")
            print(f"Test tasks: {len(test_tasks)}")
            print(f"Total tasks: {len(train_tasks) + len(test_tasks)}")
            
            # Show some example IDs
            train_list = sorted(list(train_tasks))
            test_list = sorted(list(test_tasks))
            
            print(f"\nTraining task ID range: {min(train_list)} - {max(train_list)}")
            print(f"Test task ID range: {min(test_list)} - {max(test_list)}")
            
            print(f"\nFirst 10 training task IDs: {train_list[:10]}")
            print(f"First 10 test task IDs: {test_list[:10]}")
        
        print("="*60)

def main():
    """Test the task selector"""
    try:
        with TaskSelector(props.DATABASE_PATH) as selector:
            selector.print_split_summary()
    except Exception as e:
        print(f"[ERROR] Task selector test failed: {e}")
        raise

if __name__ == "__main__":
    main()