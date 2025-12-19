#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Big Experiment Runner
Systematically tests all combinations of key parameters while optimizing embedding reuse
"""

import os
import sys
import csv
import subprocess
import time
from datetime import datetime
from itertools import product
from typing import List, Dict, Any, Tuple
import shutil
import properties as props
from sci_log import sci_log

# Define all parameter combinations
EXPERIMENT_PARAMETERS = {
    'VECTORISER_MODEL': ['own', 'fast_text', 'glove', 'bert', 'llm'],
    'DISTANCE_METRICS': [['cosine'], ['euclidean'], ['manhattan'], ['cosine', 'euclidean', 'manhattan']],
    'MODULE_VECTOR_STRATEGY': ['avg', 'sum', 'median', 'weighted_avg', 'cluster'],
    'PREPROCESS_TEST_TASK': [True, False],
    'EXCLUDE_TEST_TASKS_FROM_MODEL': [True, False],
    'NORMALIZE_VECTORS': [True, False]
}

# Parameters that affect embeddings (require CLEAR_EMBEDDINGS=True when changed)
EMBEDDING_AFFECTING_PARAMS = {
    'VECTORISER_MODEL',
    'EXCLUDE_TEST_TASKS_FROM_MODEL',
    'NORMALIZE_VECTORS'
}

class BigExperimentRunner:
    def __init__(self):
        self.results_dir = "big_experiment_results"
        self.csv_file = os.path.join(self.results_dir, "experiment_summary.csv")
        self.current_experiment = 0
        self.total_experiments = 0
        self.start_time = None
        
        # Track current embedding configuration to optimize reuse
        self.current_embedding_config = {}
        
    def setup_results_directory(self):
        """Create results directory structure"""
        if os.path.exists(self.results_dir):
            backup_dir = f"{self.results_dir}_backup_{int(time.time())}"
            shutil.move(self.results_dir, backup_dir)
            print(f"Backed up existing results to {backup_dir}")
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "statistical_outputs"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "sci_logs"), exist_ok=True)
        
        print(f"Created results directory: {self.results_dir}")
    
    def calculate_total_experiments(self) -> int:
        """Calculate total number of experiment combinations"""
        total = 1
        for param_values in EXPERIMENT_PARAMETERS.values():
            total *= len(param_values)
        return total
    
    def create_csv_header(self):
        """Create CSV file with headers"""
        headers = [
            'experiment_id',
            'timestamp',
            'vectoriser_model',
            'distance_metrics',
            'module_vector_strategy', 
            'preprocess_test_task',
            'exclude_test_tasks_from_model',
            'normalize_vectors',
            'embeddings_reused',
            'main_success',
            'vectoriser_success',
            'evaluator_success',
            'total_duration_seconds',
            'vectoriser_duration_seconds',
            'evaluator_duration_seconds',
            'map_mean',
            'map_std',
            'mrr_mean', 
            'mrr_std',
            'recall_5_mean',
            'recall_10_mean',
            'recall_20_mean',
            'recall_50_mean',
            'notes'
        ]
        
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        
        print(f"Created CSV summary file: {self.csv_file}")
    
    def embedding_config_changed(self, config: Dict[str, Any]) -> bool:
        """Check if embedding-affecting parameters have changed"""
        for param in EMBEDDING_AFFECTING_PARAMS:
            if param in config and config[param] != self.current_embedding_config.get(param):
                return True
        return False
    
    def update_properties(self, config: Dict[str, Any], clear_embeddings: bool):
        """Update properties.py with experiment configuration"""
        # Update all parameters
        for param, value in config.items():
            setattr(props, param, value)
        
        # Set CLEAR_EMBEDDINGS based on whether embeddings need regeneration
        props.CLEAR_EMBEDDINGS = clear_embeddings
        
        # Ensure VERBOSE is enabled for progress tracking
        props.VERBOSE = True
    
    def run_pipeline_step(self, script_name: str, output_file: str = None) -> Tuple[bool, float, str]:
        """Run a single pipeline step and return success status, duration, and output"""
        start_time = time.time()
        
        try:
            if output_file:
                # Redirect output to file
                with open(output_file, 'w', encoding='utf-8') as f:
                    result = subprocess.run(
                        [sys.executable, script_name],
                        stdout=f,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=3600  # 1 hour timeout
                    )
            else:
                # Capture output
                result = subprocess.run(
                    [sys.executable, script_name],
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                return True, duration, result.stdout
            else:
                error_msg = f"Error in {script_name}: {result.stderr}"
                print(f"  ❌ {script_name} failed: {result.stderr[:200]}...")
                return False, duration, error_msg
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            error_msg = f"Timeout in {script_name} after 1 hour"
            print(f"  ⏰ {script_name} timed out after 1 hour")
            return False, duration, error_msg
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Exception in {script_name}: {str(e)}"
            print(f"  💥 {script_name} exception: {str(e)}")
            return False, duration, error_msg
    
    def extract_statistics_from_output(self, output_file: str) -> Dict[str, float]:
        """Extract key statistics from statistical_evaluator.py output"""
        stats = {}
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for statistics patterns in the output
            lines = content.split('\n')
            current_metric = None
            
            for line in lines:
                line = line.strip()
                
                # Detect metric sections
                if "Mean Average Precision (MAP) Statistics:" in line:
                    current_metric = "MAP"
                elif "Mean Reciprocal Rank (MRR) Statistics:" in line:
                    current_metric = "MRR"
                elif "Recall@5 Statistics:" in line:
                    current_metric = "Recall@5"
                elif "Recall@10 Statistics:" in line:
                    current_metric = "Recall@10"
                elif "Recall@20 Statistics:" in line:
                    current_metric = "Recall@20"
                elif "Recall@50 Statistics:" in line:
                    current_metric = "Recall@50"
                
                # Extract mean and std values
                if current_metric and "Mean" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            value = float(parts[1])
                            stats[f"{current_metric.lower().replace('@', '_')}_mean"] = value
                        except (ValueError, IndexError):
                            pass
                
                if current_metric and "Std Dev" in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            value = float(parts[2])
                            stats[f"{current_metric.lower().replace('@', '_')}_std"] = value
                        except (ValueError, IndexError):
                            pass
        
        except Exception as e:
            print(f"  ⚠️  Error extracting statistics: {e}")
        
        return stats
    
    def run_single_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment configuration"""
        exp_start_time = time.time()
        
        # Determine if we need to clear embeddings
        clear_embeddings = self.embedding_config_changed(config)
        
        if clear_embeddings:
            print(f"  🔄 Embedding configuration changed, will regenerate embeddings")
        else:
            print(f"  ♻️  Reusing existing embeddings")
        
        # Update properties
        self.update_properties(config, clear_embeddings)
        
        # Create experiment-specific filenames
        exp_id = f"exp_{self.current_experiment:04d}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        statistical_output = os.path.join(
            self.results_dir, "statistical_outputs", 
            f"{exp_id}_{timestamp}_statistical_output.txt"
        )
        
        results = {
            'experiment_id': exp_id,
            'timestamp': timestamp,
            'vectoriser_model': config['VECTORISER_MODEL'],
            'distance_metrics': '_'.join(config['DISTANCE_METRICS']),
            'module_vector_strategy': config['MODULE_VECTOR_STRATEGY'],
            'preprocess_test_task': config['PREPROCESS_TEST_TASK'],
            'exclude_test_tasks_from_model': config['EXCLUDE_TEST_TASKS_FROM_MODEL'],
            'normalize_vectors': config['NORMALIZE_VECTORS'],
            'embeddings_reused': not clear_embeddings,
            'main_success': False,
            'vectoriser_success': False,
            'evaluator_success': False,
            'total_duration_seconds': 0,
            'vectoriser_duration_seconds': 0,
            'evaluator_duration_seconds': 0,
            'notes': ''
        }
        
        # Start sci_log for this experiment
        sci_log.start(__file__)
        sci_log.key('experiment_id', exp_id)
        sci_log.key('vectoriser_model', config['VECTORISER_MODEL'])
        sci_log.key('distance_metrics', '_'.join(config['DISTANCE_METRICS']))
        sci_log.key('module_vector_strategy', config['MODULE_VECTOR_STRATEGY'])
        sci_log.key('preprocess_test_task', config['PREPROCESS_TEST_TASK'])
        sci_log.key('exclude_test_tasks_from_model', config['EXCLUDE_TEST_TASKS_FROM_MODEL'])
        sci_log.key('normalize_vectors', config['NORMALIZE_VECTORS'])
        sci_log.key('embeddings_reused', not clear_embeddings)
        
        try:
            # Step 1: Run main.py (only if embeddings are being regenerated or first run)
            if clear_embeddings or self.current_experiment == 1:
                print(f"  📊 Running main.py...")
                main_success, main_duration, main_output = self.run_pipeline_step('main.py')
                results['main_success'] = main_success
                sci_log.key('main_success', main_success)
                sci_log.key('main_duration', main_duration)
                
                if not main_success:
                    results['notes'] += f"main.py failed; "
                    raise Exception("main.py failed")
            else:
                print(f"  ⏭️  Skipping main.py (embeddings reused)")
                results['main_success'] = True
            
            # Step 2: Run vectoriser.py
            print(f"  🔢 Running vectoriser.py...")
            vectoriser_success, vectoriser_duration, vectoriser_output = self.run_pipeline_step('vectoriser.py')
            results['vectoriser_success'] = vectoriser_success
            results['vectoriser_duration_seconds'] = vectoriser_duration
            sci_log.key('vectoriser_success', vectoriser_success)
            sci_log.key('vectoriser_duration', vectoriser_duration)
            
            if not vectoriser_success:
                results['notes'] += f"vectoriser.py failed; "
                raise Exception("vectoriser.py failed")
            
            # Step 3: Run statistical_evaluator.py
            print(f"  📈 Running statistical_evaluator.py...")
            evaluator_success, evaluator_duration, evaluator_output = self.run_pipeline_step(
                'statistical_evaluator.py', statistical_output
            )
            results['evaluator_success'] = evaluator_success
            results['evaluator_duration_seconds'] = evaluator_duration
            sci_log.key('evaluator_success', evaluator_success)
            sci_log.key('evaluator_duration', evaluator_duration)
            
            if evaluator_success:
                # Extract statistics from output
                stats = self.extract_statistics_from_output(statistical_output)
                results.update(stats)
                
                # Log key statistics
                for key, value in stats.items():
                    sci_log.key(key, value)
                
                print(f"  ✅ Experiment completed successfully")
                if 'map_mean' in stats:
                    print(f"     MAP: {stats['map_mean']:.4f}")
                if 'mrr_mean' in stats:
                    print(f"     MRR: {stats['mrr_mean']:.4f}")
            else:
                results['notes'] += f"statistical_evaluator.py failed; "
                raise Exception("statistical_evaluator.py failed")
            
        except Exception as e:
            results['notes'] += f"Exception: {str(e)}; "
            print(f"  ❌ Experiment failed: {str(e)}")
        
        finally:
            # Calculate total duration
            results['total_duration_seconds'] = time.time() - exp_start_time
            sci_log.key('total_duration', results['total_duration_seconds'])
            
            # Stop sci_log
            sci_log.stop('csv')
            
            # Move sci_log files to experiment directory
            try:
                import glob
                log_files = glob.glob('log/key_*.csv') + glob.glob('log/script_*.py')
                for log_file in log_files:
                    if os.path.exists(log_file):
                        dest = os.path.join(self.results_dir, "sci_logs", f"{exp_id}_{os.path.basename(log_file)}")
                        shutil.copy2(log_file, dest)
            except Exception as e:
                print(f"  ⚠️  Error copying sci_log files: {e}")
        
        # Update current embedding configuration if this experiment succeeded
        if results['vectoriser_success']:
            for param in EMBEDDING_AFFECTING_PARAMS:
                self.current_embedding_config[param] = config[param]
        
        return results
    
    def save_result_to_csv(self, result: Dict[str, Any]):
        """Save experiment result to CSV file"""
        # Define the order of fields to match CSV header
        fields = [
            'experiment_id', 'timestamp', 'vectoriser_model', 'distance_metrics',
            'module_vector_strategy', 'preprocess_test_task', 'exclude_test_tasks_from_model',
            'normalize_vectors', 'embeddings_reused', 'main_success', 'vectoriser_success',
            'evaluator_success', 'total_duration_seconds', 'vectoriser_duration_seconds',
            'evaluator_duration_seconds', 'map_mean', 'map_std', 'mrr_mean', 'mrr_std',
            'recall_5_mean', 'recall_10_mean', 'recall_20_mean', 'recall_50_mean', 'notes'
        ]
        
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            row = [result.get(field, '') for field in fields]
            writer.writerow(row)
    
    def generate_experiment_combinations(self) -> List[Dict[str, Any]]:
        """Generate all experiment combinations, ordered to maximize embedding reuse"""
        # Get all combinations
        param_names = list(EXPERIMENT_PARAMETERS.keys())
        param_values = list(EXPERIMENT_PARAMETERS.values())
        
        all_combinations = []
        for combination in product(*param_values):
            config = dict(zip(param_names, combination))
            all_combinations.append(config)
        
        # Sort combinations to group by embedding-affecting parameters
        # This maximizes reuse of expensive embeddings
        def sort_key(config):
            embedding_key = tuple(config[param] for param in sorted(EMBEDDING_AFFECTING_PARAMS))
            other_key = tuple(str(config[param]) for param in sorted(param_names) if param not in EMBEDDING_AFFECTING_PARAMS)
            return (embedding_key, other_key)
        
        all_combinations.sort(key=sort_key)
        
        print(f"Generated {len(all_combinations)} experiment combinations")
        print(f"Ordered to maximize embedding reuse across {len(EMBEDDING_AFFECTING_PARAMS)} embedding-affecting parameters")
        
        return all_combinations
    
    def run_all_experiments(self):
        """Run all experiment combinations"""
        print("🚀 Starting Big Experiment Runner")
        print("="*80)
        
        # Setup
        self.setup_results_directory()
        combinations = self.generate_experiment_combinations()
        self.total_experiments = len(combinations)
        self.create_csv_header()
        
        print(f"Total experiments to run: {self.total_experiments}")
        print(f"Results will be saved to: {self.results_dir}")
        print(f"Summary CSV: {self.csv_file}")
        print("="*80)
        
        # Track timing
        self.start_time = time.time()
        successful_experiments = 0
        
        # Run experiments
        for i, config in enumerate(combinations, 1):
            self.current_experiment = i
            elapsed = time.time() - self.start_time
            
            print(f"\n[{i}/{self.total_experiments}] Experiment {i}")
            print(f"Elapsed: {elapsed/60:.1f}min | ETA: {(elapsed/i)*(self.total_experiments-i)/60:.1f}min")
            print(f"Config: {config['VECTORISER_MODEL']} | {config['MODULE_VECTOR_STRATEGY']} | " +
                  f"norm={config['NORMALIZE_VECTORS']} | split={config['EXCLUDE_TEST_TASKS_FROM_MODEL']}")
            
            result = self.run_single_experiment(config)
            self.save_result_to_csv(result)
            
            if result['evaluator_success']:
                successful_experiments += 1
            
            # Print progress summary
            success_rate = successful_experiments / i * 100
            print(f"Progress: {i}/{self.total_experiments} ({i/self.total_experiments*100:.1f}%) | " +
                  f"Success rate: {success_rate:.1f}%")
        
        # Final summary
        total_duration = time.time() - self.start_time
        print(f"\n🎉 Big Experiment completed!")
        print(f"Total time: {total_duration/3600:.2f} hours")
        print(f"Successful experiments: {successful_experiments}/{self.total_experiments}")
        print(f"Results saved to: {self.csv_file}")
        print(f"Individual outputs in: {self.results_dir}")

def main():
    """Main entry point"""
    try:
        runner = BigExperimentRunner()
        runner.run_all_experiments()
    except KeyboardInterrupt:
        print("\n🛑 Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Experiment failed with error: {e}")
        raise

if __name__ == "__main__":
    main()