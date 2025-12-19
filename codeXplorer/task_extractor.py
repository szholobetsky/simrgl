"""
Task Extractor module for extracting task identifiers from commit messages.
"""

import re
import logging
from typing import Optional


class TaskExtractor:
    """Extracts task identifiers from commit messages using regex patterns."""

    def __init__(self, task_pattern: str):
        """
        Initialize the task extractor.

        Args:
            task_pattern: Regex pattern to match task identifiers in commit messages
        """
        self.task_pattern = task_pattern
        self.logger = logging.getLogger(__name__)

    def extract_task_name(self, message: str) -> Optional[str]:
        """
        Extract task identifier from a commit message.

        Args:
            message: Commit message text

        Returns:
            Task name if found, None otherwise
        """
        if not message:
            return None

        match = re.match(self.task_pattern, message)
        if match:
            task_name = match.group(0)

            # Handle bracketed format like [JIRA-123]
            if task_name.startswith('[') and task_name.endswith(']'):
                task_name = self._extract_from_brackets(task_name)

            return task_name

        return None

    def _extract_from_brackets(self, text: str) -> Optional[str]:
        """
        Extract task identifier from bracketed text.

        Args:
            text: Text containing task identifier in brackets

        Returns:
            Task identifier without brackets, or None if not found
        """
        pattern = r"\[([A-Z]+-\d+)\]"
        match = re.search(pattern, text)

        if match:
            return match.group(1)

        return None

    def process_all_commits(self, db_manager):
        """
        Process all commits in the database and extract task names.

        Args:
            db_manager: DatabaseManager instance for database operations
        """
        self.logger.info("Starting task extraction from commit messages")

        # Get all commits
        commits = db_manager.get_commits_without_task()
        extracted_count = 0

        for commit_id, message in commits:
            task_name = self.extract_task_name(message)

            if task_name:
                db_manager.update_task_name_in_rawdata(commit_id, task_name)
                extracted_count += 1

        self.logger.info(f"Extracted {extracted_count} task names from {len(commits)} commits")

        # Get unique task names and insert into TASK table
        task_names = db_manager.get_distinct_task_names()
        self.logger.info(f"Found {len(task_names)} unique tasks")

        for task_name in task_names:
            db_manager.insert_task(task_name)

        self.logger.info("Task extraction completed")
