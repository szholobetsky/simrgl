"""
Git Connector for extracting commit information from local repositories.
"""

import git
import logging
from typing import Optional
from tqdm import tqdm


class GitConnector:
    """Handles extraction of commit data from a Git repository."""

    def __init__(self, repo_path: str):
        """
        Initialize the Git connector.

        Args:
            repo_path: Path to the local Git repository
        """
        self.repo_path = repo_path
        self.logger = logging.getLogger(__name__)

    def extract_commits(self, db_manager, branch: str = 'master',
                       test_mode: bool = False, print_content: bool = False):
        """
        Extract commit details from the repository and store them in the database.

        Args:
            db_manager: DatabaseManager instance for storing data
            branch: Git branch to extract commits from (default: 'master')
            test_mode: If True, limits extraction to 100,000 commits for testing
            print_content: If True, prints commit details to console
        """
        try:
            repo = git.Repo(self.repo_path)
            commits = list(repo.iter_commits(branch))
            counter = 0

            self.logger.info(f"Found {len(commits)} commits in branch '{branch}'")

            for commit in tqdm(commits, desc="Processing commits", unit="commit"):
                sha = commit.hexsha
                author_name = commit.author.name
                author_email = commit.author.email
                date = commit.committed_datetime.isoformat()
                message = commit.message

                # Get diffs for this commit
                parent = commit.parents[0] if commit.parents else git.NULL_TREE
                diffs = commit.diff(parent, create_patch=True)

                for diff in tqdm(diffs, desc="Processing files", unit="file",
                               total=len(diffs), leave=False):
                    counter += 1
                    path = diff.a_path or diff.b_path or ""

                    # Decode diff content
                    try:
                        diff_content = diff.diff.decode('utf-8', errors='ignore') if diff.diff else ""
                    except Exception as e:
                        self.logger.warning(f"Could not decode diff for {path}: {e}")
                        diff_content = ""

                    # Store in database
                    if not test_mode:
                        db_manager.insert_commit_data(
                            counter, sha, author_name, author_email,
                            date, message, path, diff_content
                        )

                    # Print if requested
                    if print_content:
                        self._print_commit_details(
                            counter, sha, author_name, author_email,
                            date, message, path, diff_content
                        )

                    # Break early in test mode
                    if test_mode and counter > 100000:
                        self.logger.info("Test mode: Stopped after 100,000 commits")
                        return

            self.logger.info(f"Successfully processed {counter} commit entries")

        except git.exc.GitError as e:
            self.logger.error(f"Git error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error extracting commits: {e}")
            raise

    def _print_commit_details(self, counter: int, sha: str, author_name: str,
                             author_email: str, date: str, message: str,
                             path: str, diff: str):
        """
        Print commit details to console.

        Args:
            counter: Commit counter
            sha: Git commit SHA
            author_name: Author's name
            author_email: Author's email
            date: Commit date
            message: Commit message
            path: File path
            diff: Git diff content
        """
        print("-" * 80)
        print(f"COUNTER = {counter}")
        print(f"Commit: {sha}")
        print(f"Author: {author_name} <{author_email}>")
        print(f"Date: {date}")
        print(f"Message: {message}")
        print("Changed files:")
        print(f"File: {path}")
        print(diff)
