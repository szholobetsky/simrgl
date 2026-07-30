#!/usr/bin/env python3
"""
Merge every per-(project, task_unit) comprehensive_results.csv into one
master dataset for cross-project comparison.

Safe/idempotent to run at any point, not just once everything finishes -
partial results are still useful (see EXPERIMENT_PLAN.md Sec.6).
"""

import argparse
import glob
import os

import pandas as pd


def collect_results(results_dir: str) -> pd.DataFrame:
    pattern = os.path.join(results_dir, '*', '*', 'comprehensive_results.csv')
    frames = []

    for csv_path in sorted(glob.glob(pattern)):
        parts = csv_path.split(os.sep)
        # .../<results_dir>/<project>/<task_unit>/comprehensive_results.csv
        task_unit = parts[-2]
        project = parts[-3]

        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        df.insert(0, 'task_unit', task_unit)
        df.insert(0, 'project', project)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', default='experiment_results',
                         help='Root directory containing <project>/<task_unit>/comprehensive_results.csv')
    parser.add_argument('--output', default=None,
                         help='Output CSV path (default: <results-dir>/all_projects_results.csv)')
    args = parser.parse_args()

    output_file = args.output or os.path.join(args.results_dir, 'all_projects_results.csv')

    combined = collect_results(args.results_dir)
    if combined.empty:
        print(f"No comprehensive_results.csv files found under {args.results_dir}/*/*/")
        return

    combined.to_csv(output_file, index=False)

    print(f"Wrote {len(combined)} row(s) from "
          f"{combined[['project', 'task_unit']].drop_duplicates().shape[0]} (project, task_unit) run(s) "
          f"to {output_file}")
    for (project, task_unit), group in combined.groupby(['project', 'task_unit']):
        print(f"  {project}/{task_unit}: {len(group)} rows")


if __name__ == '__main__':
    main()
