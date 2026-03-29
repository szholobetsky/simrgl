"""
GitHub API Connector for fetching issue details from GitHub repositories.
Uses the GitHub REST API v3 — same interface as JiraApiConnector.
"""

import time
import logging
import requests
from typing import Tuple


class GitHubApiConnector:
    """
    Fetches GitHub issue details using the GitHub REST API.

    Implements the same interface as JiraApiConnector:
        fetch_task_details(issue_number: str) -> (title, description, comments)

    Authentication:
        Without a token: 60 requests/hour (not enough for large projects).
        With a personal access token: 5000 requests/hour.
        Create one at: https://github.com/settings/tokens
        Required scopes: none (public repos), or 'repo' (private repos).
    """

    BASE_URL = "https://api.github.com"

    def __init__(self, owner: str, repo: str, token: str = None):
        """
        Initialize the GitHub API connector.

        Args:
            owner: Repository owner (username or organization), e.g. 'django'
            repo:  Repository name, e.g. 'django'
            token: GitHub personal access token (optional but strongly recommended)
        """
        self.owner = owner
        self.repo = repo
        self.base = f"{self.BASE_URL}/repos/{owner}/{repo}"
        self.logger = logging.getLogger(__name__)

        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def fetch_task_details(self, issue_number: str) -> Tuple[str, str, str]:
        """
        Fetch issue details from GitHub.

        Args:
            issue_number: Issue number as a string, e.g. '4521'

        Returns:
            Tuple of (title, description, comments).
            Returns ('', '', '') on any error or if the number points to a PR.
        """
        url = f"{self.base}/issues/{issue_number}"

        try:
            response = requests.get(url, headers=self.headers, timeout=30)

            if response.status_code == 404:
                self.logger.debug(f"Issue #{issue_number} not found (404)")
                return "", "", ""

            if response.status_code == 410:
                self.logger.debug(f"Issue #{issue_number} deleted (410)")
                return "", "", ""

            if response.status_code == 403:
                reset_ts = response.headers.get("X-RateLimit-Reset")
                if reset_ts:
                    sleep_secs = max(0, int(reset_ts) - int(time.time())) + 5
                    self.logger.warning(
                        f"GitHub rate limit hit. Sleeping {sleep_secs}s until reset."
                    )
                    time.sleep(sleep_secs)
                    # Retry once after sleeping
                    response = requests.get(url, headers=self.headers, timeout=30)
                    if response.status_code != 200:
                        return "", "", ""
                    data = response.json()
                else:
                    self.logger.warning(
                        f"GitHub rate limit hit. Consider adding a GITHUB_TOKEN to config.py."
                    )
                    return "", "", ""

            response.raise_for_status()
            data = response.json()

            title = data.get("title", "") or ""
            description = data.get("body", "") or ""  # plain Markdown, cleaner than Jira ADF

            comments = self._fetch_comments(data)

            return title, description, comments

        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout fetching issue #{issue_number}")
            return "", "", ""
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching issue #{issue_number}: {e}")
            return "", "", ""
        except Exception as e:
            self.logger.error(f"Unexpected error fetching issue #{issue_number}: {e}")
            return "", "", ""

    def _fetch_comments(self, issue_data: dict) -> str:
        """
        Fetch all comments for an issue and join them into a single string.

        Args:
            issue_data: Parsed JSON response for the issue

        Returns:
            All comment bodies joined by a space, or '' if no comments.
        """
        if issue_data.get("comments", 0) == 0:
            return ""

        comments_url = issue_data.get("comments_url", "")
        if not comments_url:
            return ""

        try:
            response = requests.get(comments_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            comments_data = response.json()
            return " ".join(c.get("body", "") or "" for c in comments_data)
        except Exception as e:
            self.logger.warning(f"Could not fetch comments: {e}")
            return ""

    def check_rate_limit(self) -> dict:
        """
        Check current GitHub API rate limit status.

        Returns:
            Dict with 'remaining', 'limit', 'reset_in_seconds' keys.
            Useful for estimating how long a large fetch will take.
        """
        try:
            response = requests.get(
                f"{self.BASE_URL}/rate_limit",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            core = response.json().get("resources", {}).get("core", {})
            remaining = core.get("remaining", 0)
            limit = core.get("limit", 0)
            reset_ts = core.get("reset", 0)
            reset_in = max(0, reset_ts - int(time.time()))
            return {
                "remaining": remaining,
                "limit": limit,
                "reset_in_seconds": reset_in,
            }
        except Exception as e:
            self.logger.warning(f"Could not check rate limit: {e}")
            return {"remaining": -1, "limit": -1, "reset_in_seconds": -1}
