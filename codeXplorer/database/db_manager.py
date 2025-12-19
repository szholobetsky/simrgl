"""
Database Manager for CodeXplorer.
Handles SQLite database creation, table management, and data operations.
"""

import sqlite3
import logging
from typing import Optional, List, Tuple


class DatabaseManager:
    """Manages SQLite database operations for commit and task data."""

    def __init__(self, db_file: str):
        """
        Initialize the database manager.

        Args:
            db_file: Path to the SQLite database file
        """
        self.db_file = db_file
        self.logger = logging.getLogger(__name__)

    def create_tables(self):
        """Create necessary database tables if they don't exist."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()

            # Create RAWDATA table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS RAWDATA (
                    ID INTEGER PRIMARY KEY,
                    SHA TEXT,
                    AUTHOR_NAME TEXT,
                    AUTHOR_EMAIL TEXT,
                    CMT_DATE TEXT,
                    MESSAGE BLOB,
                    PATH BLOB,
                    DIFF BLOB,
                    TASK_NAME TEXT
                )
            """)

            # Create TASK table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS TASK (
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    NAME TEXT UNIQUE,
                    TITLE TEXT,
                    DESCRIPTION TEXT,
                    COMMENTS TEXT
                )
            """)

            # Create indexes for TASK table
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS TASK_ID_INDX
                ON TASK (ID ASC)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS TASK_NAME_INDX
                ON TASK (NAME ASC)
            """)

            conn.commit()
            self.logger.info("Database tables created successfully")

        except sqlite3.Error as error:
            self.logger.error(f"Error creating tables: {error}")
            raise
        finally:
            if conn:
                conn.close()

    def insert_commit_data(self, commit_id: int, sha: str, author_name: str,
                          author_email: str, date: str, message: str,
                          path: str, diff: str):
        """
        Insert commit data into the RAWDATA table.

        Args:
            commit_id: Unique identifier for the commit
            sha: Git commit SHA
            author_name: Author's name
            author_email: Author's email
            date: Commit date
            message: Commit message
            path: File path
            diff: Git diff content
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO RAWDATA
                (ID, SHA, AUTHOR_NAME, AUTHOR_EMAIL, CMT_DATE, MESSAGE, PATH, DIFF)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (commit_id, sha, author_name, author_email, date, message, path, diff))

            conn.commit()

        except sqlite3.Error as error:
            self.logger.error(f"Error inserting commit data: {error}")
            raise
        finally:
            if conn:
                conn.close()

    def insert_commit_data_batch(self, data_list: List[Tuple]):
        """
        Insert multiple commit records in batches for better performance.

        Args:
            data_list: List of tuples containing commit data
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()

            batch_size = 100
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i+batch_size]
                cursor.executemany("""
                    INSERT INTO RAWDATA
                    (ID, SHA, AUTHOR_NAME, AUTHOR_EMAIL, CMT_DATE, MESSAGE, PATH, DIFF)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, batch)
                conn.commit()

            self.logger.info(f"Inserted {len(data_list)} commit records in batches")

        except sqlite3.Error as error:
            self.logger.error(f"Error inserting batch data: {error}")
            raise
        finally:
            if conn:
                conn.close()

    def update_task_name_in_rawdata(self, commit_id: int, task_name: str):
        """
        Update the task name for a specific commit in RAWDATA.

        Args:
            commit_id: ID of the commit to update
            task_name: Task name extracted from commit message
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE RAWDATA
                SET TASK_NAME = ?
                WHERE ID = ?
            """, (task_name, commit_id))

            conn.commit()

        except sqlite3.Error as error:
            self.logger.error(f"Error updating task name: {error}")
            raise
        finally:
            if conn:
                conn.close()

    def get_commits_without_task(self) -> List[Tuple]:
        """
        Get all commits that don't have a task name assigned.

        Returns:
            List of tuples containing (id, message)
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()

            cursor.execute("SELECT ID, MESSAGE FROM RAWDATA")
            rows = cursor.fetchall()
            return rows

        except sqlite3.Error as error:
            self.logger.error(f"Error fetching commits: {error}")
            raise
        finally:
            if conn:
                conn.close()

    def insert_task(self, task_name: str):
        """
        Insert a new task into the TASK table.

        Args:
            task_name: Unique task identifier (e.g., JIRA-123)
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO TASK (NAME)
                VALUES (?)
            """, (task_name,))

            conn.commit()

        except sqlite3.IntegrityError:
            # Task already exists, skip
            pass
        except sqlite3.Error as error:
            self.logger.error(f"Error inserting task: {error}")
            raise
        finally:
            if conn:
                conn.close()

    def get_tasks_without_details(self) -> List[str]:
        """
        Get task names that don't have title, description, or comments.

        Returns:
            List of task names
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT NAME
                FROM TASK
                WHERE TITLE IS NULL AND NAME IS NOT NULL
            """)

            return [row[0] for row in cursor.fetchall()]

        except sqlite3.Error as error:
            self.logger.error(f"Error fetching tasks: {error}")
            raise
        finally:
            if conn:
                conn.close()

    def update_task_details(self, task_name: str, title: str,
                           description: str, comments: str):
        """
        Update task details with information from Jira.

        Args:
            task_name: Task identifier
            title: Task title/summary
            description: Task description
            comments: All task comments concatenated
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE TASK
                SET TITLE = ?, DESCRIPTION = ?, COMMENTS = ?
                WHERE NAME = ?
            """, (title, description, comments, task_name))

            conn.commit()

        except sqlite3.Error as error:
            self.logger.error(f"Error updating task details: {error}")
            raise
        finally:
            if conn:
                conn.close()

    def get_distinct_task_names(self) -> List[str]:
        """
        Get all distinct task names from RAWDATA.

        Returns:
            List of unique task names
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT DISTINCT TASK_NAME
                FROM RAWDATA
                WHERE TASK_NAME IS NOT NULL
                ORDER BY TASK_NAME ASC
            """)

            return [row[0] for row in cursor.fetchall()]

        except sqlite3.Error as error:
            self.logger.error(f"Error fetching distinct task names: {error}")
            raise
        finally:
            if conn:
                conn.close()
