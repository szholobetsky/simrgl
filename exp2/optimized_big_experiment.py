#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Big Experiment Runner
Focused experiment comparing embedding models with different aggregation strategies
"""

import os
import sys
import csv
import subprocess
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
import shutil
import properties as props
from sci_log import sci_log
from properties_updater import update_experiment_properties, backup_original_properties, restore_original_properties

# Define the focused experiment parameters
EXPERIMENT_MODELS = [
    # Model 1: Word2Vec (own)
    {
        'VECTORISER_MODEL': 'own',
        'model_description': 'Word2Vec (custom trained)'
    },
    
    # Model 2: FastText
    {
        'VECTORISER_MODEL': 'fast_text',
        'model_description': 'FastText (pretrained)'
    },
    
    # Model 3: GloVe
    {
        'VECTORISER_MODEL': 'glove',
        'GLOVE_MODEL_NAME': 'glove.6B.100d',
        'model_description': 'GloVe (6B.100d)'
    },
    
    # Model 4: BERT base
    {
        'VECTORISER_MODEL': 'bert',
        'BERT_MODEL_NAME': 'bert-base-uncased',
        'model_description': 'BERT (base-uncased)'
    },
    
    # Model 5: RoBERTa
    {
        'VECTORISER_MODEL': 'bert',
        'BERT_MODEL_NAME': 'roberta-base',
        'model_description': 'RoBERTa (base)'
    },
    
    # Model 6: CodeBERT
    {
        'VECTORISER_MODEL': 'bert',
        'BERT_MODEL_NAME': 'microsoft/codebert-base',
        'model_description': 'CodeBERT (microsoft)'
    },
    
    # Model 7: MPNet (sentence-transformers default)
    {
        'VECTORISER_MODEL': 'llm',
        'LLM_MODEL_NAME': 'sentence-transformers/all-mpnet-base-v2',
        'model_description': 'MPNet (all-mpnet-base-v2)'
    },
    
    # Model 8: BGE Large
    {
        'VECTORISER_MODEL': 'llm',
        'LLM_MODEL_NAME': 'BAAI/bge-large-en-v1.5',
        'model_description': 'BGE (large-en-v1.5)'
    },
    
    # Model 9: MS MARCO DistilBERT
    {
        'VECTORISER_MODEL': 'llm',
        'LLM_MODEL_NAME': 'sentence-transformers/msmarco-distilbert-base-v4',
        'model_description': 'MS MARCO (distilbert-base-v4)'
    }
]

# Module aggregation strategies to test
MODULE_STRATEGIES = ['avg', 'sum', 'median', 'weighted_avg', 'cluster']

# Fixed parameters for all experiments
FIXED_PARAMS = {
    'NORMALIZE_VECTORS': True,
    'EXCLUDE_TEST_TASKS_FROM_MODEL': True,
    'PREPROCESS_TEST_TASK': False,
    'DISTANCE_METRICS': ['cosine'],
    'CLEAR_EMBEDDINGS': True,  # Will be managed automatically
}

class OptimizedExperimentRunner:
    def __init__(self):
        self.results_dir = "focused_experiment_results"
        self.csv_file = os.path.join(self.results_dir, "model_comparison_summary.csv")
        self.current_experiment = 0
        self.total_experiments = len(EXPERIMENT_MODELS) * len(MODULE_STRATEGIES)
        self.start_time = None
        
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
    
    def create_csv_header(self):
        """Create CSV file with headers"""
        headers = [
            'experiment_id',
            'model_group',
            'model_description', 
            'vectoriser_model',
            'specific_model_name',
            'module_strategy',
            'embeddings_rebuilt',
            'main_success',
            'vectoriser_success',
            'evaluator_success',
            'total_duration_minutes',
            'vectoriser_duration_minutes',
            'evaluator_duration_minutes',
            'map_mean',
            'map_std',
            'mrr_mean',
            'mrr_std',
            'recall_5_mean',
            'recall_10_mean',
            'recall_20_mean',
            'recall_50_mean',
            'timestamp',
            'notes'
        ]
        
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        
        print(f"Created CSV summary file: {self.csv_file}")
    
    def update_properties(self, model_config: Dict[str, Any], strategy: str, clear_embeddings: bool):
        """Update properties.py file with experiment configuration"""
        update_experiment_properties(model_config, strategy, clear_embeddings, FIXED_PARAMS)


    # def update_properties(self, model_config: Dict[str, Any], strategy: str, clear_embeddings: bool):
    #     """Update properties.py with experiment configuration"""
    #     # Apply fixed parameters
    #     for param, value in FIXED_PARAMS.items():
    #         setattr(props, param, value)
        
    #     # Apply model-specific parameters
    #     for param, value in model_config.items():
    #         if param != 'model_description':  # Skip description field
    #             setattr(props, param, value)
        
    #     # Set strategy
    #     props.MODULE_VECTOR_STRATEGY = strategy
        
    #     # Set embedding rebuild flag
    #     props.CLEAR_EMBEDDINGS = clear_embeddings
        
    #     # Ensure verbose mode
    #     props.VERBOSE = True
    
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
                        timeout=7200  # 2 hour timeout
                    )
            else:
                # Capture output
                result = subprocess.run(
                    [sys.executable, script_name],
                    capture_output=True,
                    text=True,
                    timeout=7200  # 2 hour timeout
                )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                return True, duration, result.stdout
            else:
                error_msg = f"Error in {script_name}: {result.stderr}"
                print(f"  [x] {script_name} failed:")
                print(f"     FULL ERROR OUTPUT:")
                print(f"     {'-'*50}")
                print(result.stderr)
                print(f"     {'-'*50}")
                return False, duration, error_msg
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            error_msg = f"Timeout in {script_name} after 2 hours"
            print(f"  [o] {script_name} timed out after 2 hours")
            return False, duration, error_msg
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Exception in {script_name}: {str(e)}"
            print(f"  [><] {script_name} exception:")
            print(f"     FULL EXCEPTION:")
            print(f"     {'-'*50}")
            import traceback
            traceback.print_exc()
            print(f"     {'-'*50}")
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
            print(f"  [ ! ]  Error extracting statistics: {e}")
        
        return stats
    
    def run_single_experiment(self, model_config: Dict[str, Any], strategy: str, 
                            model_index: int, strategy_index: int) -> Dict[str, Any]:
        """Run a single experiment configuration"""
        exp_start_time = time.time()
        
        # Determine if we need to rebuild embeddings (first strategy for each model)
        rebuild_embeddings = (strategy_index == 0)
        
        # Create experiment ID
        exp_id = f"M{model_index+1:02d}S{strategy_index+1:02d}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_description = model_config.get('model_description', 'Unknown')
        specific_model = model_config.get('BERT_MODEL_NAME') or model_config.get('LLM_MODEL_NAME') or 'default'
        
        print(f"\n[{self.current_experiment+1}/{self.total_experiments}] Experiment {exp_id}")
        print(f"Model: {model_description}")
        print(f"Strategy: {strategy}")
        print(f"Rebuild embeddings: {rebuild_embeddings}")
        
        # Update properties
        self.update_properties(model_config, strategy, rebuild_embeddings)
        
        results = {
            'experiment_id': exp_id,
            'model_group': model_index + 1,
            'model_description': model_description,
            'vectoriser_model': model_config['VECTORISER_MODEL'],
            'specific_model_name': specific_model,
            'module_strategy': strategy,
            'embeddings_rebuilt': rebuild_embeddings,
            'main_success': False,
            'vectoriser_success': False,
            'evaluator_success': False,
            'total_duration_minutes': 0,
            'vectoriser_duration_minutes': 0,
            'evaluator_duration_minutes': 0,
            'timestamp': timestamp,
            'notes': ''
        }
        
        # Start sci_log for this experiment
        sci_log.start(__file__)
        sci_log.key('experiment_id', exp_id)
        sci_log.key('model_description', model_description)
        sci_log.key('vectoriser_model', model_config['VECTORISER_MODEL'])
        sci_log.key('specific_model_name', specific_model)
        sci_log.key('module_strategy', strategy)
        sci_log.key('embeddings_rebuilt', rebuild_embeddings)
        
        statistical_output = os.path.join(
            self.results_dir, "statistical_outputs", 
            f"{exp_id}_{timestamp}_output.txt"
        )
        
        rebuild_main = False # Should be true only when we change task excluding strategy
        try:
            # Step 1: Run main.py (only for first strategy of each model)
            if rebuild_main:
                print(f"  << Running main.py...")
                main_success, main_duration, main_output = self.run_pipeline_step('main.py')
                results['main_success'] = main_success
                sci_log.key('main_success', main_success)
                sci_log.key('main_duration', main_duration)
                
                if not main_success:
                    results['notes'] += f"main.py failed; "
                    raise Exception("main.py failed")
            else:
                print(f"  [...]  Skipping main.py (reusing existing terms)")
                results['main_success'] = True
            
            # Step 2: Run vectorizer.py
            if rebuild_embeddings:
                print(f"  # Running vectorizer.py...")
                vectoriser_success, vectoriser_duration, vectoriser_output = self.run_pipeline_step('vectorizer.py')
                results['vectoriser_success'] = vectoriser_success
                results['vectoriser_duration_minutes'] = vectoriser_duration / 60
                sci_log.key('vectoriser_success', vectoriser_success)
                sci_log.key('vectoriser_duration', vectoriser_duration)
                
                if not vectoriser_success:
                    results['notes'] += f"vectoriser.py failed; "
                    raise Exception("vectoriser.py failed")
            
            # Step 3: Run statistical_evaluator.py
            print(f"  / Running statistical_evaluator.py...")
            evaluator_success, evaluator_duration, evaluator_output = self.run_pipeline_step(
                'statistical_evaluator.py', statistical_output
            )
            results['evaluator_success'] = evaluator_success
            results['evaluator_duration_minutes'] = evaluator_duration / 60
            sci_log.key('evaluator_success', evaluator_success)
            sci_log.key('evaluator_duration', evaluator_duration)
            
            if evaluator_success:
                # Extract statistics from output
                stats = self.extract_statistics_from_output(statistical_output)
                results.update(stats)
                
                # Log key statistics
                for key, value in stats.items():
                    sci_log.key(key, value)
                
                print(f"  [v] Experiment completed successfully")
                if 'map_mean' in stats:
                    print(f"     MAP: {stats['map_mean']:.4f}")
                if 'mrr_mean' in stats:
                    print(f"     MRR: {stats['mrr_mean']:.4f}")
            else:
                results['notes'] += f"statistical_evaluator.py failed; "
                raise Exception("statistical_evaluator.py failed")
            
        except Exception as e:
            results['notes'] += f"Exception: {str(e)}; "
            print(f"  [x] Experiment failed: {str(e)}")
        
        finally:
            # Calculate total duration
            results['total_duration_minutes'] = (time.time() - exp_start_time) / 60
            sci_log.key('total_duration', results['total_duration_minutes'] * 60)
            
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
                print(f"  [ ! ]  Error copying sci_log files: {e}")
        
        return results
    
    def save_result_to_csv(self, result: Dict[str, Any]):
        """Save experiment result to CSV file"""
        fields = [
            'experiment_id', 'model_group', 'model_description', 'vectoriser_model',
            'specific_model_name', 'module_strategy', 'embeddings_rebuilt',
            'main_success', 'vectoriser_success', 'evaluator_success',
            'total_duration_minutes', 'vectoriser_duration_minutes', 'evaluator_duration_minutes',
            'map_mean', 'map_std', 'mrr_mean', 'mrr_std',
            'recall_5_mean', 'recall_10_mean', 'recall_20_mean', 'recall_50_mean',
            'timestamp', 'notes'
        ]
        
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            row = [result.get(field, '') for field in fields]
            writer.writerow(row)
    
    def run_all_experiments(self):
        """Run all model and strategy combinations"""
        print(">>> Starting Focused Model Comparison Experiment")
        print("="*80)
        
        # Setup
        self.setup_results_directory()
        self.create_csv_header()
        
        print(f"Models to test: {len(EXPERIMENT_MODELS)}")
        print(f"Strategies per model: {len(MODULE_STRATEGIES)}")
        print(f"Total experiments: {self.total_experiments}")
        print(f"Embedding rebuilds: {len(EXPERIMENT_MODELS)}")
        print(f"Results will be saved to: {self.results_dir}")
        print("="*80)
        
        # Track timing
        self.start_time = time.time()
        successful_experiments = 0
        
        # Run experiments
        for model_index, model_config in enumerate(EXPERIMENT_MODELS):
            print(f"\n<<< STARTING MODEL GROUP {model_index + 1}/{len(EXPERIMENT_MODELS)}")
            print(f"Model: {model_config['model_description']}")
            print("-" * 60)
            
            for strategy_index, strategy in enumerate(MODULE_STRATEGIES):
                result = self.run_single_experiment(
                    model_config, strategy, model_index, strategy_index
                )
                self.save_result_to_csv(result)
                
                if result['evaluator_success']:
                    successful_experiments += 1
                
                self.current_experiment += 1
                
                # Print progress
                elapsed = time.time() - self.start_time
                eta = (elapsed / self.current_experiment) * (self.total_experiments - self.current_experiment)
                success_rate = successful_experiments / self.current_experiment * 100
                
                print(f"Progress: {self.current_experiment}/{self.total_experiments} | "
                      f"Success: {success_rate:.1f}% | ETA: {eta/60:.1f}min")
        
        # Final summary
        total_duration = time.time() - self.start_time
        print(f"\n[!!!] Focused Experiment completed!")
        print(f"Total time: {total_duration/3600:.2f} hours")
        print(f"Successful experiments: {successful_experiments}/{self.total_experiments}")
        print(f"Results saved to: {self.csv_file}")

def main():
    """Main entry point"""
    try:
        backup_original_properties()
        runner = OptimizedExperimentRunner()
        try:
            runner.run_all_experiments()
        finally:
            # Always restore original properties when done
            restore_original_properties()
    except KeyboardInterrupt:
        print("\n[-] Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[><] Experiment failed with error: {e}")
        raise

if __name__ == "__main__":
    main()