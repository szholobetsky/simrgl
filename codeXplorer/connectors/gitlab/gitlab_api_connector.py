"""
GitLab API Connector for fetching issue details from GitLab instances.
Uses the GitLab REST API v4 — same interface as JiraApiConnector.

Verified (2026-03):
  - Issues endpoint:  public, no token needed
  - Comments (notes): REQUIRES token even for public projects on gitlab.com
  - Rate limit:       500 req/10min unauthenticated (~3000/hr)
  - With token:       2000 req/min (~120,000/hr)

IMPORTANT — commit link rate warning:
  GitLab projects typically do NOT reference issues in commit messages.
  Issue linking is done via Merge Requests (MR closing patterns).
  Measured link rates (2026-03):
    Inkscape:    ~2%  (unusable with commit regex)
    GNOME Shell: ~1%  (unusable with commit regex)
    Mesa:        ~6%  (unusable with commit regex)

  For GitLab projects, use COMMIT_SOURCE = 'mr' (Merge Request based)
  instead of commit message regex. See fetch_mr_issue_links() below.

Supported GitLab instances:
  gitlab.com                  — public projects
  gitlab.gnome.org            — GNOME projects
  gitlab.freedesktop.org      — Mesa, Wayland, D-Bus
  Any self-hosted GitLab      — set base_url accordingly
"""

import logging
import requests
from typing import Tuple, List, Dict, Optional


class GitLabApiConnector:
    """
    Fetches GitLab issue details using the GitLab REST API v4.

    Implements the same interface as JiraApiConnector:
        fetch_task_details(issue_iid: str) -> (title, description, comments)

    Note on project identification:
        GitLab uses numeric project IDs or 'namespace/project' slugs.
        Example: 'inkscape/inkscape' or '1234567'
        URL-encode the slug: 'inkscape%2Finkscape'
    """

    def __init__(self, base_url: str, project: str, token: str = None):
        """
        Initialize the GitLab API connector.

        Args:
            base_url: Base URL of the GitLab instance, e.g.:
                      'https://gitlab.com'
                      'https://gitlab.gnome.org'
                      'https://gitlab.freedesktop.org'
            project:  Project path as 'namespace/repo', e.g. 'inkscape/inkscape'
                      or numeric project ID as string, e.g. '1234567'
            token:    Personal access token.
                      Required for fetching comments (notes) on gitlab.com.
                      Generate at: GitLab → Settings → Access Tokens → api scope.
        """
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.logger = logging.getLogger(__name__)

        # URL-encode the project path (replace / with %2F)
        if '/' in project and not project.isdigit():
            self.project_id = project.replace('/', '%2F')
        else:
            self.project_id = project

        self.api_base = f"{self.base_url}/api/v4/projects/{self.project_id}"

        self.headers = {"Accept": "application/json"}
        if token:
            self.headers["PRIVATE-TOKEN"] = token

    def fetch_task_details(self, issue_iid: str) -> Tuple[str, str, str]:
        """
        Fetch issue details from GitLab.

        Args:
            issue_iid: Issue internal ID (iid) as string, e.g. '1234'
                       Note: GitLab has two IDs — global 'id' and per-project 'iid'.
                       Commit messages always reference 'iid' (#NNN within the project).

        Returns:
            Tuple of (title, description, comments).
            Returns ('', '', '') on any error.
        """
        url = f"{self.api_base}/issues/{issue_iid}"

        try:
            response = requests.get(url, headers=self.headers, timeout=30)

            if response.status_code == 404:
                self.logger.debug(f"Issue #{issue_iid} not found (404)")
                return "", "", ""

            if response.status_code == 401:
                self.logger.warning(
                    f"GitLab auth required for issue #{issue_iid}. "
                    f"Set GITLAB_TOKEN in config.py."
                )
                return "", "", ""

            response.raise_for_status()
            data = response.json()

            title = data.get("title", "") or ""
            description = data.get("description", "") or ""  # plain Markdown

            comments = self._fetch_comments(issue_iid)

            return title, description, comments

        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout fetching issue #{issue_iid}")
            return "", "", ""
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching issue #{issue_iid}: {e}")
            return "", "", ""
        except Exception as e:
            self.logger.error(f"Unexpected error fetching issue #{issue_iid}: {e}")
            return "", "", ""

    def _fetch_comments(self, issue_iid: str) -> str:
        """
        Fetch human-written comments (notes) for an issue.
        Filters out system notes (e.g. 'closed via merge request').

        Requires token on gitlab.com — returns '' without one.
        """
        if not self.token:
            return ""

        url = f"{self.api_base}/issues/{issue_iid}/notes"
        params = {"per_page": 100, "sort": "asc"}

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            if response.status_code in (401, 403):
                return ""
            response.raise_for_status()
            notes = response.json()
            return " ".join(
                n.get("body", "") or ""
                for n in notes
                if not n.get("system", False) and n.get("body")
            )
        except Exception as e:
            self.logger.warning(f"Could not fetch comments for issue #{issue_iid}: {e}")
            return ""

    # -------------------------------------------------------------------------
    # MR-based issue linking (use instead of commit regex for GitLab projects)
    # -------------------------------------------------------------------------

    def fetch_mr_issue_links(self, max_mrs: int = 5000) -> List[Dict]:
        """
        Fetch merged MRs and their linked issues — the reliable way to get
        task→file links for GitLab projects that don't use issue refs in commits.

        GitLab MR closing pattern: "Closes #NNN" in MR description → links issue.

        Returns:
            List of dicts: [{'issue_iid': '123', 'commits': ['sha1', ...], 'files': ['path', ...]}, ...]

        Usage:
            links = connector.fetch_mr_issue_links()
            # Then store in DB manually — this bypasses the standard pipeline.
        """
        self.logger.info(f"Fetching merged MRs (max {max_mrs})...")
        result = []
        page = 1

        while len(result) < max_mrs:
            url = f"{self.api_base}/merge_requests"
            params = {"state": "merged", "per_page": 100, "page": page}

            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
                response.raise_for_status()
                mrs = response.json()
            except Exception as e:
                self.logger.error(f"Error fetching MRs page {page}: {e}")
                break

            if not mrs:
                break

            for mr in mrs:
                mr_iid = mr.get("iid")
                closing_issues = self._fetch_mr_closing_issues(mr_iid)
                if not closing_issues:
                    continue

                commits = self._fetch_mr_commits(mr_iid)
                files = self._fetch_mr_files(mr_iid)

                for issue_iid in closing_issues:
                    result.append({
                        "issue_iid": str(issue_iid),
                        "mr_iid": str(mr_iid),
                        "commits": commits,
                        "files": files,
                    })

            page += 1
            if len(mrs) < 100:
                break

        self.logger.info(f"Found {len(result)} issue→MR links")
        return result

    def _fetch_mr_closing_issues(self, mr_iid: int) -> List[int]:
        """Returns list of issue iids closed by this MR."""
        url = f"{self.api_base}/merge_requests/{mr_iid}/closes_issues"
        try:
            r = requests.get(url, headers=self.headers, timeout=30)
            if r.status_code != 200:
                return []
            return [issue["iid"] for issue in r.json()]
        except Exception:
            return []

    def _fetch_mr_commits(self, mr_iid: int) -> List[str]:
        """Returns list of commit SHAs in this MR."""
        url = f"{self.api_base}/merge_requests/{mr_iid}/commits"
        try:
            r = requests.get(url, headers=self.headers, timeout=30)
            if r.status_code != 200:
                return []
            return [c["id"] for c in r.json()]
        except Exception:
            return []

    def _fetch_mr_files(self, mr_iid: int) -> List[str]:
        """Returns list of file paths changed in this MR."""
        url = f"{self.api_base}/merge_requests/{mr_iid}/diffs"
        try:
            r = requests.get(url, headers=self.headers, params={"per_page": 100}, timeout=30)
            if r.status_code != 200:
                return []
            return [d["new_path"] for d in r.json() if d.get("new_path")]
        except Exception:
            return []
