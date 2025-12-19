"""
Jira API Connector for fetching task details using Jira's REST API.
"""

import requests
import logging
from typing import Tuple


class JiraApiConnector:
    """Fetches Jira ticket details using the official Jira REST API."""

    def __init__(self, jira_url: str):
        """
        Initialize the Jira API connector.

        Args:
            jira_url: Base URL of the Jira instance (e.g., https://issues.apache.org/jira)
        """
        self.jira_url = jira_url.rstrip('/')
        self.logger = logging.getLogger(__name__)

    def fetch_task_details(self, task_key: str) -> Tuple[str, str, str]:
        """
        Fetch task details from Jira using the REST API.

        Args:
            task_key: Jira task key (e.g., JIRA-123)

        Returns:
            Tuple of (title, description, comments)
        """
        url = f"{self.jira_url}/rest/api/3/issue/{task_key}"
        headers = {'Accept': 'application/json'}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()

            title = data.get('fields', {}).get('summary', '')
            description = str(data.get('fields', {}).get('description', ''))

            # Extract all comments
            comments_data = data.get('fields', {}).get('comment', {}).get('comments', [])
            comments = str([comment.get('body', '') for comment in comments_data])

            return title, description, comments

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching task {task_key}: {e}")
            return "", "", ""
        except Exception as e:
            self.logger.error(f"Unexpected error fetching task {task_key}: {e}")
            return "", "", ""
