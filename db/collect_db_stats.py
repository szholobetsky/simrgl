"""
Collect per-project statistics from each CodeXplorer SQLite database.

Scans a directory for *.db files (default: ../data) and, for each one,
computes metrics needed to decide which "task unit" criterion is usable
for the RAG retrieval experiment (exp3 / exp3.1):

  - ticket criterion: TASK table rows, linked via RAWDATA.TASK_NAME
  - commit criterion: individual commits (grouped by SHA), using the
    commit MESSAGE as a TITLE/DESCRIPTION surrogate

Output: a CSV with one row per project database.
"""

import argparse
import csv
import glob
import os
import sqlite3


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    )
    return cur.fetchone() is not None


def collect_stats(db_path: str) -> dict:
    project = os.path.splitext(os.path.basename(db_path))[0]
    conn = sqlite3.connect(db_path)

    stats = {
        "project": project,
        "db_path": db_path,
        "db_size_mb": round(os.path.getsize(db_path) / (1024 * 1024), 2),
    }

    # Schema note: exp3's data/sonar.db uses RAWDATA; the project DBs built
    # more recently by codeXplorer (Projects/db/*.db) use COMMITS instead.
    # Support both so this script works across both generations.
    commits_table = "COMMITS" if table_exists(conn, "COMMITS") else "RAWDATA"
    has_rawdata = table_exists(conn, commits_table)
    has_task = table_exists(conn, "TASK")
    stats["commits_table_name"] = commits_table if has_rawdata else None
    stats["has_rawdata_table"] = has_rawdata
    stats["has_task_table"] = has_task

    if has_rawdata:
        total_rows = conn.execute(f"SELECT COUNT(*) FROM {commits_table}").fetchone()[0]
        distinct_commits = conn.execute(
            f"SELECT COUNT(DISTINCT SHA) FROM {commits_table}"
        ).fetchone()[0]
        linked_rows = conn.execute(
            f"SELECT COUNT(*) FROM {commits_table} WHERE TASK_NAME IS NOT NULL AND TASK_NAME != ''"
        ).fetchone()[0]
        linked_commits = conn.execute(
            f"SELECT COUNT(DISTINCT SHA) FROM {commits_table} "
            "WHERE TASK_NAME IS NOT NULL AND TASK_NAME != ''"
        ).fetchone()[0]
        distinct_linked_tasks = conn.execute(
            f"SELECT COUNT(DISTINCT TASK_NAME) FROM {commits_table} "
            "WHERE TASK_NAME IS NOT NULL AND TASK_NAME != ''"
        ).fetchone()[0]
        date_range = conn.execute(
            f"SELECT MIN(CMT_DATE), MAX(CMT_DATE) FROM {commits_table}"
        ).fetchone()

        stats["rawdata_rows"] = total_rows
        stats["distinct_commits"] = distinct_commits
        stats["avg_files_per_commit"] = (
            round(total_rows / distinct_commits, 2) if distinct_commits else 0
        )
        stats["commits_with_task_name"] = linked_commits
        stats["linkage_rate_pct"] = (
            round(100 * linked_commits / distinct_commits, 2) if distinct_commits else 0
        )
        stats["distinct_linked_tasks"] = distinct_linked_tasks
        stats["cmt_date_min"] = date_range[0]
        stats["cmt_date_max"] = date_range[1]
    else:
        stats.update({
            "rawdata_rows": 0, "distinct_commits": 0, "avg_files_per_commit": 0,
            "commits_with_task_name": 0, "linkage_rate_pct": 0,
            "distinct_linked_tasks": 0, "cmt_date_min": None, "cmt_date_max": None,
        })

    if has_task:
        total_tasks = conn.execute("SELECT COUNT(*) FROM TASK").fetchone()[0]
        tasks_with_desc = conn.execute(
            "SELECT COUNT(*) FROM TASK WHERE DESCRIPTION IS NOT NULL AND DESCRIPTION != ''"
        ).fetchone()[0]
        tasks_with_comments = conn.execute(
            "SELECT COUNT(*) FROM TASK WHERE COMMENTS IS NOT NULL AND COMMENTS != ''"
        ).fetchone()[0]
        stats["task_rows"] = total_tasks
        stats["tasks_with_description"] = tasks_with_desc
        stats["tasks_with_comments"] = tasks_with_comments
    else:
        stats["task_rows"] = 0
        stats["tasks_with_description"] = 0
        stats["tasks_with_comments"] = 0

    # Feasibility flags for a 200-task held-out test set (need train volume too,
    # so require roughly 3x the test size as a safety margin -> 600).
    min_needed = 600
    stats["ticket_criterion_usable"] = stats["distinct_linked_tasks"] >= min_needed
    stats["commit_criterion_usable"] = stats["distinct_commits"] >= min_needed
    stats["comparable_ticket_vs_commit"] = (
        stats["ticket_criterion_usable"] and has_task
    )

    conn.close()
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data"),
        help="Directory to scan for *.db files",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "project_stats.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    db_files = sorted(glob.glob(os.path.join(args.data_dir, "*.db")))
    if not db_files:
        print(f"No .db files found in {args.data_dir}")
        return

    rows = [collect_stats(db) for db in db_files]

    fieldnames = list(rows[0].keys())
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} row(s) to {args.output}")
    for r in rows:
        print(f"  {r['project']}: commits={r['distinct_commits']}, "
              f"linked_tasks={r['distinct_linked_tasks']}, "
              f"linkage_rate={r['linkage_rate_pct']}%")


if __name__ == "__main__":
    main()
