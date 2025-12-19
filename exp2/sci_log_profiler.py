import os
import json
import csv
import statistics
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import subprocess
import sys


class SciLogProfiler:
    def __init__(self, log_dir: str = "log"):
        self.log_dir = log_dir
        self.experiments_data: Dict[int, Dict[str, Any]] = {}
        self.metrics_stats: Dict[str, Dict[str, float]] = {}
        
    def scan_experiments(self) -> None:
        """Scan all experiment files in the log directory and collect data."""
        if not os.path.exists(self.log_dir):
            print(f"Log directory '{self.log_dir}' not found!")
            return
            
        print(f"Scanning experiments in '{self.log_dir}'...")
        
        # Get all key files (both CSV and JSON)
        key_files = []
        for filename in os.listdir(self.log_dir):
            if filename.startswith("key_") and (filename.endswith(".csv") or filename.endswith(".json")):
                key_files.append(filename)
        
        if not key_files:
            print("No experiment data files found!")
            return
            
        # Load data from each file
        for filename in sorted(key_files):
            run_number = self._extract_run_number(filename)
            if run_number is not None:
                data = self._load_experiment_data(filename)
                if data:
                    self.experiments_data[run_number] = data
        
        print(f"Loaded {len(self.experiments_data)} experiments")
        
    def _extract_run_number(self, filename: str) -> Optional[int]:
        """Extract run number from filename."""
        try:
            # Remove 'key_' prefix and file extension
            if filename.startswith("key_"):
                num_part = filename[4:]  # Remove 'key_'
                if num_part.endswith('.csv'):
                    num_part = num_part[:-4]
                elif num_part.endswith('.json'):
                    num_part = num_part[:-5]
                return int(num_part)
        except ValueError:
            pass
        return None
    
    def _load_experiment_data(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load experiment data from CSV or JSON file."""
        filepath = os.path.join(self.log_dir, filename)
        
        try:
            if filename.endswith('.csv'):
                return self._load_csv(filepath)
            elif filename.endswith('.json'):
                return self._load_json(filepath)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None
    
    def _load_csv(self, filepath: str) -> Dict[str, Any]:
        """Load data from CSV file."""
        data = {}
        with open(filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    key, value = row[0], row[1]
                    # Try to convert to appropriate type
                    data[key] = self._convert_value(value)
        return data
    
    def _load_json(self, filepath: str) -> Dict[str, Any]:
        """Load data from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as jsonfile:
            return json.load(jsonfile)
    
    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type."""
        # Try int
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Keep as string
        return value
    
    def calculate_statistics(self) -> None:
        """Calculate statistics for all numeric metrics."""
        if not self.experiments_data:
            print("No experiment data loaded!")
            return
            
        print("Calculating statistics...")
        
        # Collect all numeric metrics
        numeric_metrics = defaultdict(list)
        
        for run_num, data in self.experiments_data.items():
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    numeric_metrics[key].append(value)
        
        # Calculate statistics for each metric
        for metric, values in numeric_metrics.items():
            if len(values) > 0:
                self.metrics_stats[metric] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'avg': statistics.mean(values),
                    'median': statistics.median(values),
                    'stdev': statistics.stdev(values) if len(values) > 1 else 0.0
                }
    
    def display_statistics(self) -> None:
        """Display comprehensive statistics table."""
        if not self.metrics_stats:
            print("No statistics calculated!")
            return
            
        print("\n" + "="*80)
        print("EXPERIMENT STATISTICS SUMMARY")
        print("="*80)
        
        # Header
        print(f"{'Metric':<20} {'Count':<6} {'Min':<12} {'Max':<12} {'Avg':<12} {'Median':<12} {'St.Dev':<12}")
        print("-" * 80)
        
        # Sort metrics for consistent display
        for metric in sorted(self.metrics_stats.keys()):
            stats = self.metrics_stats[metric]
            print(f"{metric:<20} {stats['count']:<6} "
                  f"{stats['min']:<12.3f} {stats['max']:<12.3f} "
                  f"{stats['avg']:<12.3f} {stats['median']:<12.3f} "
                  f"{stats['stdev']:<12.3f}")
    
    def find_best_experiments(self, metric: str, minimize: bool = True, top_n: int = 5) -> List[Tuple[int, float]]:
        """Find best experiments for a specific metric."""
        if not self.experiments_data:
            return []
            
        results = []
        for run_num, data in self.experiments_data.items():
            if metric in data and isinstance(data[metric], (int, float)):
                results.append((run_num, data[metric]))
        
        if not results:
            return []
            
        # Sort based on minimize/maximize preference
        results.sort(key=lambda x: x[1], reverse=not minimize)
        return results[:top_n]
    
    def display_leaderboard(self, metric: str, minimize: bool = True, top_n: int = 5) -> None:
        """Display leaderboard for a specific metric (LeetCode style)."""
        best_experiments = self.find_best_experiments(metric, minimize, top_n)
        
        if not best_experiments:
            print(f"No data found for metric '{metric}'")
            return
            
        print(f"\n{'ðŸ†' if minimize else 'ðŸ“ˆ'} TOP {top_n} EXPERIMENTS - {metric.upper()}")
        print(f"{'Minimizing' if minimize else 'Maximizing'} {metric}")
        print("="*60)
        
        for i, (run_num, value) in enumerate(best_experiments, 1):
            medal = "ðŸ¥‡" if i == 1 else "ðŸ¥ˆ" if i == 2 else "ðŸ¥‰" if i == 3 else f"{i}."
            print(f"{medal} Run #{run_num:03d}: {value:.4f}")
            
            # Show additional context for top 3
            if i <= 3:
                exp_data = self.experiments_data[run_num]
                print(f"   â””â”€ Duration: {exp_data.get('diff', 'N/A'):.3f}s, "
                      f"Started: {exp_data.get('start', 'N/A')}")
    
    def show_experiment_code(self, run_number: int) -> None:
        """Display the code for a specific experiment."""
        script_filename = f"script_{run_number:03d}.py"
        script_path = os.path.join(self.log_dir, script_filename)
        
        if not os.path.exists(script_path):
            print(f"Script file not found: {script_path}")
            return
            
        print(f"\nðŸ“„ CODE FOR EXPERIMENT #{run_number:03d}")
        print("="*60)
        
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Add line numbers
            for i, line in enumerate(lines, 1):
                print(f"{i:3d} | {line}", end='')
                
        except Exception as e:
            print(f"Error reading script file: {e}")
    
    def compare_experiments(self, run_numbers: List[int]) -> None:
        """Compare specific experiments side by side."""
        if len(run_numbers) < 2:
            print("Need at least 2 experiments to compare")
            return
            
        print(f"\nâš–ï¸  EXPERIMENT COMPARISON")
        print("="*80)
        
        # Get common metrics
        common_metrics = set()
        valid_runs = []
        
        for run_num in run_numbers:
            if run_num in self.experiments_data:
                valid_runs.append(run_num)
                if not common_metrics:
                    common_metrics = set(self.experiments_data[run_num].keys())
                else:
                    common_metrics &= set(self.experiments_data[run_num].keys())
        
        if not valid_runs:
            print("No valid experiments found for comparison")
            return
            
        # Display comparison table
        print(f"{'Metric':<20}", end='')
        for run_num in valid_runs:
            print(f"{'Run #' + str(run_num):>15}", end='')
        print()
        print("-" * (20 + 15 * len(valid_runs)))
        
        for metric in sorted(common_metrics):
            if all(isinstance(self.experiments_data[run][metric], (int, float)) 
                   for run in valid_runs):
                print(f"{metric:<20}", end='')
                
                values = [self.experiments_data[run][metric] for run in valid_runs]
                best_idx = values.index(min(values))  # Assuming lower is better
                
                for i, run_num in enumerate(valid_runs):
                    value = self.experiments_data[run_num][metric]
                    marker = " ðŸ†" if i == best_idx else ""
                    print(f"{value:>13.3f}{marker:>2}", end='')
                print()
    
    def interactive_mode(self) -> None:
        """Interactive mode for exploring experiments."""
        while True:
            print(f"\nðŸ”¬ SCI-LOG PROFILER - Interactive Mode")
            print("="*50)
            print("1. Show statistics summary")
            print("2. Show leaderboard for metric")
            print("3. View experiment code")
            print("4. Compare experiments")
            print("5. List all experiments")
            print("6. Rescan experiments")
            print("0. Exit")
            
            choice = input("\nEnter your choice (0-6): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self.display_statistics()
            elif choice == '2':
                self._interactive_leaderboard()
            elif choice == '3':
                self._interactive_view_code()
            elif choice == '4':
                self._interactive_compare()
            elif choice == '5':
                self._list_all_experiments()
            elif choice == '6':
                self.scan_experiments()
                self.calculate_statistics()
            else:
                print("Invalid choice!")
    
    def _interactive_leaderboard(self) -> None:
        """Interactive leaderboard selection."""
        if not self.metrics_stats:
            print("No statistics available!")
            return
            
        print(f"\nAvailable metrics: {', '.join(sorted(self.metrics_stats.keys()))}")
        metric = input("Enter metric name: ").strip()
        
        if metric not in self.metrics_stats:
            print(f"Metric '{metric}' not found!")
            return
            
        minimize_input = input("Minimize this metric? (y/n, default=y): ").strip().lower()
        minimize = minimize_input != 'n'
        
        try:
            top_n = int(input("How many top results? (default=5): ") or "5")
        except ValueError:
            top_n = 5
            
        self.display_leaderboard(metric, minimize, top_n)
    
    def _interactive_view_code(self) -> None:
        """Interactive code viewing."""
        try:
            run_number = int(input("Enter experiment run number: "))
            self.show_experiment_code(run_number)
        except ValueError:
            print("Invalid run number!")
    
    def _interactive_compare(self) -> None:
        """Interactive experiment comparison."""
        runs_input = input("Enter run numbers separated by commas: ")
        try:
            run_numbers = [int(x.strip()) for x in runs_input.split(',')]
            self.compare_experiments(run_numbers)
        except ValueError:
            print("Invalid run numbers!")
    
    def _list_all_experiments(self) -> None:
        """List all available experiments."""
        if not self.experiments_data:
            print("No experiments found!")
            return
            
        print(f"\nðŸ“‹ ALL EXPERIMENTS ({len(self.experiments_data)} total)")
        print("="*60)
        
        for run_num in sorted(self.experiments_data.keys()):
            data = self.experiments_data[run_num]
            duration = data.get('diff', 'N/A')
            start_time = data.get('start', 'N/A')
            print(f"Run #{run_num:03d}: Duration={duration:.3f}s, Started={start_time}")


def main():
    """Main function with command line interface."""
    profiler = SciLogProfiler()
    
    if len(sys.argv) == 1:
        # Interactive mode
        profiler.scan_experiments()
        profiler.calculate_statistics()
        profiler.interactive_mode()
    else:
        # Command line mode
        command = sys.argv[1].lower()
        
        profiler.scan_experiments()
        profiler.calculate_statistics()
        
        if command == 'stats':
            profiler.display_statistics()
        elif command == 'leaderboard' and len(sys.argv) >= 3:
            metric = sys.argv[2]
            minimize = len(sys.argv) < 4 or sys.argv[3].lower() != 'max'
            top_n = int(sys.argv[4]) if len(sys.argv) >= 5 else 5
            profiler.display_leaderboard(metric, minimize, top_n)
        elif command == 'code' and len(sys.argv) >= 3:
            run_number = int(sys.argv[2])
            profiler.show_experiment_code(run_number)
        elif command == 'compare' and len(sys.argv) >= 4:
            run_numbers = [int(x) for x in sys.argv[2:]]
            profiler.compare_experiments(run_numbers)
        else:
            print("Usage:")
            print("  python sci_log_profiler.py                    # Interactive mode")
            print("  python sci_log_profiler.py stats              # Show statistics")
            print("  python sci_log_profiler.py leaderboard <metric> [min|max] [top_n]")
            print("  python sci_log_profiler.py code <run_number>  # View code")
            print("  python sci_log_profiler.py compare <run1> <run2> ...  # Compare runs")


if __name__ == "__main__":
    main()