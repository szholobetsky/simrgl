"""
Bulk Task Fetcher — downloads ALL issues from a tracker into the TASK table.

Unlike task_fetcher.py (which only fetches tasks already referenced in commits),
this module fetches the complete issue list regardless of commit history.
Required for fuzzy/embedding-based commit matching on low-discipline projects.

Usage:
    python bulk_task_fetcher.py

    Or from code:
        fetcher = BulkTaskFetcher()
        fetcher.run(db_manager)
"""

import time
import logging
import requests
from tqdm import tqdm

import config
from database import DatabaseManager


logger = logging.getLogger(__name__)


class BulkTaskFetcher:
    """
    Fetches all issues from the configured tracker and stores them in TASK table.
    Supports Jira, GitHub Issues, GitLab, and YouTrack.
    Uses pagination — safe for projects with 100k+ issues.
    Already-stored tasks are updated (upsert), not duplicated.
    """

    def run(self, db_manager: DatabaseManager):
        tracker = getattr(config, 'TRACKER_TYPE', 'jira')
        logger.info(f"Starting bulk task fetch — tracker: {tracker}")

        if tracker == 'jira':
            tasks = self._fetch_jira(config.JIRA_URL)
        elif tracker == 'github':
            tasks = self._fetch_github(config.GITHUB_OWNER, config.GITHUB_REPO,
                                       getattr(config, 'GITHUB_TOKEN', None))
        elif tracker == 'gitlab':
            tasks = self._fetch_gitlab(config.GITLAB_URL, config.GITLAB_PROJECT,
                                       getattr(config, 'GITLAB_TOKEN', None))
        elif tracker == 'youtrack':
            tasks = self._fetch_youtrack(config.YOUTRACK_URL,
                                         getattr(config, 'YOUTRACK_TOKEN', None))
        else:
            raise ValueError(f"Unknown tracker type: {tracker}")

        if not tasks:
            logger.warning("No tasks fetched.")
            return

        logger.info(f"Fetched {len(tasks)} tasks. Storing to DB...")
        db_manager.bulk_upsert_tasks(tasks)
        logger.info("Bulk task fetch complete.")

    # -------------------------------------------------------------------------
    # Jira
    # -------------------------------------------------------------------------

    def _fetch_jira(self, jira_url: str):
        """
        Fetch all issues for the configured Jira project using JQL search.
        Requires config.JIRA_PROJECT_KEY (e.g. 'KAFKA').
        """
        project_key = getattr(config, 'JIRA_PROJECT_KEY', None)
        if not project_key:
            raise ValueError(
                "JIRA_PROJECT_KEY not set in config.py. "
                "Add: JIRA_PROJECT_KEY = 'KAFKA'  (or SPARK, FLINK, etc.)"
            )

        base = jira_url.rstrip('/')
        url = f"{base}/rest/api/2/search"
        headers = {'Accept': 'application/json'}
        page_size = 100
        start = 0
        tasks = []

        print(f"\nFetching all Jira issues for project {project_key}...")

        while True:
            params = {
                'jql': f'project={project_key} ORDER BY id ASC',
                'startAt': start,
                'maxResults': page_size,
                'fields': 'summary,description,comment',
            }
            try:
                r = requests.get(url, headers=headers, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                logger.error(f"Jira API error at startAt={start}: {e}")
                break

            issues = data.get('issues', [])
            if not issues:
                break

            for issue in issues:
                name = issue.get('key', '')
                fields = issue.get('fields', {})
                title = fields.get('summary', '') or ''
                description = str(fields.get('description', '') or '')
                comments_data = fields.get('comment', {}).get('comments', [])
                comments = ' '.join(c.get('body', '') or '' for c in comments_data)
                tasks.append((name, title, description, comments))

            start += len(issues)
            total = data.get('total', 0)
            print(f"\r  {start}/{total} issues fetched...", end='', flush=True)

            if start >= total:
                break

            time.sleep(0.2)  # be polite to the server

        print()
        return tasks

    # -------------------------------------------------------------------------
    # GitHub
    # -------------------------------------------------------------------------

    def _fetch_github(self, owner: str, repo: str, token: str = None):
        """Fetch all issues (not PRs) from a GitHub repository."""
        headers = {"Accept": "application/vnd.github+json",
                   "X-GitHub-Api-Version": "2022-11-28"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        page = 1
        tasks = []

        print(f"\nFetching all GitHub issues for {owner}/{repo}...")

        while True:
            params = {'state': 'all', 'per_page': 100, 'page': page}
            try:
                r = requests.get(url, headers=headers, params=params, timeout=30)
                if r.status_code == 403:
                    reset = r.headers.get('X-RateLimit-Reset', '?')
                    logger.warning(f"GitHub rate limit. Reset at {reset}. Sleeping 60s...")
                    time.sleep(60)
                    continue
                r.raise_for_status()
                issues = r.json()
            except Exception as e:
                logger.error(f"GitHub API error page {page}: {e}")
                break

            if not issues:
                break

            for issue in issues:
                if 'pull_request' in issue:
                    continue  # skip PRs
                name = str(issue.get('number', ''))
                title = issue.get('title', '') or ''
                description = issue.get('body', '') or ''
                tasks.append((name, title, description, ''))
                # Note: comments fetched separately would require extra API calls.
                # For bulk fetch we skip comments to stay within rate limits.

            print(f"\r  page {page}, {len(tasks)} issues so far...", end='', flush=True)
            page += 1

            # Check if there are more pages via Link header
            if 'next' not in r.headers.get('Link', ''):
                break

            time.sleep(0.1)

        print()
        return tasks

    # -------------------------------------------------------------------------
    # GitLab
    # -------------------------------------------------------------------------

    def _fetch_gitlab(self, base_url: str, project: str, token: str = None):
        """Fetch all issues from a GitLab project."""
        headers = {"Accept": "application/json"}
        if token:
            headers["PRIVATE-TOKEN"] = token

        project_id = project.replace('/', '%2F')
        url = f"{base_url.rstrip('/')}/api/v4/projects/{project_id}/issues"
        page = 1
        tasks = []

        print(f"\nFetching all GitLab issues for {project}...")

        while True:
            params = {'per_page': 100, 'page': page, 'scope': 'all'}
            try:
                r = requests.get(url, headers=headers, params=params, timeout=30)
                if r.status_code == 401:
                    logger.error("GitLab auth required. Set GITLAB_TOKEN in config.py.")
                    break
                r.raise_for_status()
                issues = r.json()
            except Exception as e:
                logger.error(f"GitLab API error page {page}: {e}")
                break

            if not issues:
                break

            for issue in issues:
                name = str(issue.get('iid', ''))
                title = issue.get('title', '') or ''
                description = issue.get('description', '') or ''
                tasks.append((name, title, description, ''))

            total_pages = int(r.headers.get('X-Total-Pages', 0))
            print(f"\r  page {page}/{total_pages or '?'}, {len(tasks)} issues...",
                  end='', flush=True)
            page += 1

            if page > total_pages > 0:
                break

            time.sleep(0.1)

        print()
        return tasks

    # -------------------------------------------------------------------------
    # YouTrack
    # -------------------------------------------------------------------------

    def _fetch_youtrack(self, base_url: str, token: str = None):
        """
        Fetch all issues from a YouTrack project.
        Requires config.YOUTRACK_PROJECT_KEY (e.g. 'KT', 'IDEA', 'IJPL').
        """
        project_key = getattr(config, 'YOUTRACK_PROJECT_KEY', None)
        if not project_key:
            raise ValueError(
                "YOUTRACK_PROJECT_KEY not set in config.py. "
                "Add: YOUTRACK_PROJECT_KEY = 'KT'  (or IDEA, IJPL, etc.)"
            )

        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        url = f"{base_url.rstrip('/')}/api/issues"
        top = 100
        skip = 0
        tasks = []

        print(f"\nFetching all YouTrack issues for project {project_key}...")

        while True:
            params = {
                'query': f'project: {project_key}',
                'fields': 'idReadable,summary,description,comments(text)',
                '$top': top,
                '$skip': skip,
            }
            try:
                r = requests.get(url, headers=headers, params=params, timeout=30)
                r.raise_for_status()
                issues = r.json()
            except Exception as e:
                logger.error(f"YouTrack API error at skip={skip}: {e}")
                break

            if not issues:
                break

            for issue in issues:
                name = issue.get('idReadable', '')
                title = issue.get('summary', '') or ''
                description = issue.get('description', '') or ''
                comments_list = issue.get('comments', []) or []
                comments = ' '.join(
                    c.get('text', '') or '' for c in comments_list if c.get('text')
                )
                tasks.append((name, title, description, comments))

            skip += len(issues)
            print(f"\r  {skip} issues fetched...", end='', flush=True)

            if len(issues) < top:
                break

            time.sleep(0.1)

        print()
        return tasks


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s - %(message)s')

    db_manager = DatabaseManager(config.DB_FILE)
    db_manager.create_tables()
    db_manager.create_match_log_table()

    fetcher = BulkTaskFetcher()
    fetcher.run(db_manager)


if __name__ == '__main__':
    main()
