"""
Jira HTML Connector for fetching task details by parsing HTML pages.
Use this when Jira API is not available or requires authentication.
"""

import requests
from bs4 import BeautifulSoup
import logging
from typing import Tuple


class JiraHtmlConnector:
    """Fetches Jira ticket details by parsing HTML pages."""

    def __init__(self, jira_url: str):
        """
        Initialize the Jira HTML connector.

        Args:
            jira_url: Base URL of the Jira instance (e.g., https://issues.apache.org/jira)
        """
        self.jira_url = jira_url.rstrip('/')
        self.logger = logging.getLogger(__name__)

    def fetch_task_details(self, task_key: str) -> Tuple[str, str, str]:
        """
        Fetch task details from Jira by parsing HTML.

        Args:
            task_key: Jira task key (e.g., JIRA-123)

        Returns:
            Tuple of (title, description, comments)
        """
        comment_tab = '?page=com.atlassian.jira.plugin.system.issuetabpanels:comment-tabpanel'
        task_url = f"{self.jira_url}/browse/{task_key}{comment_tab}"

        try:
            response = requests.get(task_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract title
            title_element = soup.find('h1', {'id': 'summary-val'})
            title = title_element.text.strip() if title_element else ""

            # Extract description
            description_element = soup.find('div', {'id': 'description-val'})
            description = description_element.text.strip() if description_element else ""

            # Extract comments
            comment_elements = soup.find_all(class_='twixi-wrap concise actionContainer')
            comments = [comment.text.strip() for comment in comment_elements]

            return title, description, str(comments)

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching task {task_key}: {e}")
            return "", "", ""
        except Exception as e:
            self.logger.error(f"Unexpected error fetching task {task_key}: {e}")
            return "", "", ""
