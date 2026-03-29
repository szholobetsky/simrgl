"""
YouTrack API Connector for fetching issue details from JetBrains YouTrack.
Uses the YouTrack REST API — same interface as JiraApiConnector.

Verified (2026-03): youtrack.jetbrains.com is publicly accessible without a token.
Response time: ~0.48s/request. No X-RateLimit headers observed.
Estimated throughput: ~7,500 issues/hour unauthenticated.

Description format: mixed — newer issues return plain text,
older issues return HTML wrapped in <div class="wiki text prewrapped">.
This connector strips HTML automatically.
"""

import re
import logging
import requests
from typing import Tuple


class YouTrackApiConnector:
    """
    Fetches YouTrack issue details using the YouTrack REST API v3.

    Implements the same interface as JiraApiConnector:
        fetch_task_details(issue_id: str) -> (title, description, comments)

    Works for any YouTrack instance, not only JetBrains.
    Tested against: https://youtrack.jetbrains.com (public, no auth needed)

    Commit pattern in intellij-community:
        "IDEA-123456 Fix ..."  "IJPL-123 Refactor ..."  "PY-12345 ..."
        All match r'^[A-Z]+-\\d+' — same as Apache Jira SIMPLE_NUMBER_MASK.
    """

    def __init__(self, base_url: str, token: str = None):
        """
        Initialize the YouTrack API connector.

        Args:
            base_url: Base URL of the YouTrack instance, e.g.:
                      'https://youtrack.jetbrains.com'
                      'https://your-company.youtrack.cloud'
            token:    Permanent token for authentication (optional for public instances).
                      Generate at: YouTrack → Profile → Account Security → New token.
        """
        self.base_url = base_url.rstrip('/')
        self.logger = logging.getLogger(__name__)

        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def fetch_task_details(self, issue_id: str) -> Tuple[str, str, str]:
        """
        Fetch issue details from YouTrack.

        Args:
            issue_id: Issue ID in format 'PROJECT-NNNN', e.g. 'IDEA-300000', 'KT-71000'

        Returns:
            Tuple of (title, description, comments).
            Returns ('', '', '') on any error or missing issue.
        """
        url = f"{self.base_url}/api/issues/{issue_id}"
        params = {"fields": "summary,description,comments(text)"}

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)

            if response.status_code == 404:
                self.logger.debug(f"Issue {issue_id} not found (404)")
                return "", "", ""

            if response.status_code == 401:
                self.logger.error(
                    f"YouTrack authentication required for {issue_id}. "
                    f"Set YOUTRACK_TOKEN in config.py."
                )
                return "", "", ""

            response.raise_for_status()
            data = response.json()

            if "error" in data:
                self.logger.debug(f"Issue {issue_id}: {data.get('error_description', data['error'])}")
                return "", "", ""

            title = data.get("summary", "") or ""
            raw_description = data.get("description", "") or ""
            description = self._clean_text(raw_description)

            comments_list = data.get("comments", []) or []
            comments = " ".join(
                self._clean_text(c.get("text", "") or "")
                for c in comments_list
                if c.get("text")
            )

            return title, description, comments

        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout fetching issue {issue_id}")
            return "", "", ""
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching issue {issue_id}: {e}")
            return "", "", ""
        except Exception as e:
            self.logger.error(f"Unexpected error fetching issue {issue_id}: {e}")
            return "", "", ""

    def _clean_text(self, text: str) -> str:
        """
        Strip HTML tags from description/comments.
        Older YouTrack issues return HTML; newer ones return plain text.
        This handles both transparently.
        """
        if not text:
            return ""
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Decode common HTML entities
        text = (text
                .replace("&amp;", "&")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&quot;", '"')
                .replace("&#39;", "'")
                .replace("&nbsp;", " "))
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
