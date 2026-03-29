"""
Task Fetcher module for retrieving task details from Jira.
"""

import time
import logging
from tqdm import tqdm
from typing import Optional

import config
from connectors.jira import JiraApiConnector, JiraHtmlConnector, JiraSeleniumConnector
from connectors.github import GitHubApiConnector
from connectors.youtrack import YouTrackApiConnector
from connectors.gitlab import GitLabApiConnector


class TaskFetcher:
    """Fetches task details from a tracker (Jira or GitHub) and updates the database."""

    def __init__(self, jira_url: str, connector_type: str = 'api'):
        """
        Initialize the task fetcher.

        Args:
            jira_url: Base URL of the Jira instance (ignored when tracker_type='github')
            connector_type: Jira connector type ('api', 'html', 'selenium').
                            Ignored when config.TRACKER_TYPE == 'github'.
        """
        self.logger = logging.getLogger(__name__)

        tracker_type = getattr(config, 'TRACKER_TYPE', 'jira')

        if tracker_type == 'github':
            self.connector = GitHubApiConnector(
                owner=config.GITHUB_OWNER,
                repo=config.GITHUB_REPO,
                token=getattr(config, 'GITHUB_TOKEN', None),
            )
        elif tracker_type == 'youtrack':
            self.connector = YouTrackApiConnector(
                base_url=config.YOUTRACK_URL,
                token=getattr(config, 'YOUTRACK_TOKEN', None),
            )
        elif tracker_type == 'gitlab':
            self.connector = GitLabApiConnector(
                base_url=config.GITLAB_URL,
                project=config.GITLAB_PROJECT,
                token=getattr(config, 'GITLAB_TOKEN', None),
            )
        elif connector_type == 'api':
            self.connector = JiraApiConnector(jira_url)
        elif connector_type == 'html':
            self.connector = JiraHtmlConnector(jira_url)
        elif connector_type == 'selenium':
            self.connector = JiraSeleniumConnector(jira_url)
        else:
            raise ValueError(f"Unknown connector type: {connector_type}")

    def fetch_all_tasks(self, db_manager, rate_limit: int = 50,
                       batch_size: int = 10):
        """
        Fetch details for all tasks that don't have details yet.

        Args:
            db_manager: DatabaseManager instance for database operations
            rate_limit: Maximum number of requests per minute
            batch_size: Number of updates before committing to database
        """
        self.logger.info("Starting task details fetch from Jira")

        # Get tasks without details
        task_names = db_manager.get_tasks_without_details()

        if not task_names:
            self.logger.info("No tasks to fetch")
            return

        self.logger.info(f"Found {len(task_names)} tasks to fetch")

        # Rate limiting counters
        request_count = 0
        start_time = time.time()
        update_count = 0

        for task_name in tqdm(task_names, desc="Fetching tasks", unit="task"):
            # Rate limiting
            if request_count >= rate_limit:
                elapsed_time = time.time() - start_time
                if elapsed_time < 60:
                    sleep_time = 60 - elapsed_time
                    self.logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                start_time = time.time()
                request_count = 0

            try:
                # Fetch task details
                title, description, comments = self.connector.fetch_task_details(task_name)
                request_count += 1

                # Update database
                db_manager.update_task_details(task_name, title, description, comments)
                update_count += 1

                # Periodic commit for batching
                if update_count % batch_size == 0:
                    self.logger.debug(f"Processed {update_count} tasks")

            except Exception as e:
                self.logger.error(f"Error processing task {task_name}: {e}")
                continue

        self.logger.info(f"Successfully fetched details for {update_count} tasks")
