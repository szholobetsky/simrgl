#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Big Experiment Results Analyzer
Analyzes and visualizes results from comprehensive parameter sweep experiments
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os
import sys

class ExperimentAnalyzer:
    def __init__(self, csv_file: str = "big_experiment_results/experiment_summary.csv"):
        self.csv_file = csv_file
        self.df = None
        self.load_results()
    
    def load_results(self):
        """Load experiment results from CSV"""
        try:
            self.df = pd.read_csv(self.csv_file)
            print(f"Loaded {len(self.df)} experiments from {self.csv_file}")
            
            # Convert boolean columns
            bool_columns = ['preprocess_test_task', 'exclude_test_tasks_from_model', 
                          'normalize_vectors', 'embeddings_reused', 'main_success', 
                          'vectoriser_success', 'evaluator_success']
            
            for col in bool_columns:
                if col in self.df.columns:
                    self.df[col] = self.df[col].astype(bool)
            
            # Filter to successful experiments only for analysis
            successful = self.df[self.df['evaluator_success'] == True]
            print(f"Successfully completed experiments: {len(successful)}")
            
            if len(successful) == 0:
                print("Warning: No successful experiments found!")
            
        except FileNotFoundError:
            print(f"Error: Could not find results file {self.csv_file}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading results: {e}")
            sys.exit(1)
    
    def print_summary_statistics(self):
        """Print overall summary of experiments"""
        print("\n" + "="*80)
        print("BIG EXPERIMENT SUMMARY")
        print("="*80)
        
        total_experiments = len(self.df)
        successful_experiments = len(self.df[self.df['evaluator_success'] == True])
        
        print(f"Total experiments run: {total_experiments}")
        print(f"Successfully completed: {successful_experiments} ({successful_experiments/total_experiments*100:.1f}%)")
        
        if successful_experiments == 0:
            print("No successful experiments to analyze!")
            return
        
        # Success rates by model
        print(f"\nSuccess rates by vectorizer model:")
        for model in self.df['vectoriser_model'].unique():
            model_df = self.df[self.df['vectoriser_model'] == model]
            success_rate = len(model_df[model_df['evaluator_success'] == True]) / len(model_df) * 100
            print(f"  {model}: {success_rate:.1f}% ({len(model_df[model_df['evaluator_success'] == True])}/{len(model_df)})")
        
        # Timing analysis
        successful_df = self.df[self.df['evaluator_success'] == True]
        if len(successful_df) > 0:
            print(f"\nTiming analysis (successful experiments only):")
            total_time = successful_df['total_duration_seconds'].sum() / 3600
            avg_time = successful_df['total_duration_seconds'].mean() / 60
            vectoriser_time = successful_df['vectoriser_duration_seconds'].sum() / 3600
            
            print(f"  Total time: {total_time:.2f} hours")
            print(f"  Average per experiment: {avg_time:.1f} minutes")
            print(f"  Time spent on vectorization: {vectoriser_time:.2f} hours ({vectoriser_time/total_time*100:.1f}%)")
            
            # Embedding reuse effectiveness
            reused = len(successful_df[successful_df['embeddings_reused'] == True])
            print(f"  Embedding reuse rate: {reused/len(successful_df)*100:.1f}% ({reused}/{len(successful_df)})")
    
    def analyze_best_configurations(self, metric: str = 'map_mean', top_n: int = 10):
        """Analyze best performing configurations"""
        successful_df = self.df[self.df['evaluator_success'] == True]
        
        if len(successful_df) == 0 or metric not in successful_df.columns:
            print(f"Cannot analyze {metric} - no data available")
            return
        
        print(f"\n" + "="*80)
        print(f"TOP {top_n} CONFIGURATIONS BY {metric.upper()}")
        print("="*80)
        
        # Sort by metric and get top N
        top_configs = successful_df.nlargest(top_n, metric)
        
        for i, (idx, row) in enumerate(top_configs.iterrows(), 1):
            print(f"{i:2d}. {metric}: {row[metric]:.4f}")
            print(f"    Model: {row['vectoriser_model']}")
            print(f"    Strategy: {row['module_vector_strategy']}")
            print(f"    Normalize: {row['normalize_vectors']}")
            print(f"    Train/Test Split: {row['exclude_test_tasks_from_model']}")
            print(f"    Preprocess: {row['preprocess_test_task']}")
            print(f"    Distance: {row['distance_metrics']}")
            print(f"    Duration: {row['total_duration_seconds']/60:.1f} min")
            print()
    
    def compare_models(self, metric: str = 'map_mean'):
        """Compare performance across different models"""
        successful_df = self.df[self.df['evaluator_success'] == True]
        
        if len(successful_df) == 0:
            return
        
        print(f"\n" + "="*80)
        print(f"MODEL COMPARISON ({metric.upper()})")
        print("="*80)
        
        model_stats = successful_df.groupby('vectoriser_model')[metric].agg(['count', 'mean', 'std', 'min', 'max'])
        model_stats = model_stats.sort_values('mean', ascending=False)
        
        print(f"{'Model':<12} {'Count':<6} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
        print("-" * 60)
        
        for model, stats in model_stats.iterrows():
            print(f"{model:<12} {stats['count']:<6.0f} {stats['mean']:<8.4f} {stats['std']:<8.4f} {stats['min']:<8.4f} {stats['max']:<8.4f}")
    
    def analyze_parameter_impact(self, metric: str = 'map_mean'):
        """Analyze impact of each parameter on performance"""
        successful_df = self.df[self.df['evaluator_success'] == True]
        
        if len(successful_df) == 0:
            return
        
        print(f"\n" + "="*80)
        print(f"PARAMETER IMPACT ANALYSIS ({metric.upper()})")
        print("="*80)
        
        categorical_params = ['vectoriser_model', 'module_vector_strategy', 'distance_metrics']
        boolean_params = ['preprocess_test_task', 'exclude_test_tasks_from_model', 'normalize_vectors']
        
        # Analyze categorical parameters
        for param in categorical_params:
            if param in successful_df.columns:
                print(f"\n{param.upper()}:")
                param_stats = successful_df.groupby(param)[metric].agg(['count', 'mean', 'std'])
                param_stats = param_stats.sort_values('mean', ascending=False)
                
                for value, stats in param_stats.iterrows():
                    print(f"  {value:<20} Count: {stats['count']:3.0f} Mean: {stats['mean']:.4f} Std: {stats['std']:.4f}")
        
        # Analyze boolean parameters
        for param in boolean_params:
            if param in successful_df.columns:
                print(f"\n{param.upper()}:")
                param_stats = successful_df.groupby(param)[metric].agg(['count', 'mean', 'std'])
                
                for value, stats in param_stats.iterrows():
                    print(f"  {value:<8} Count: {stats['count']:3.0f} Mean: {stats['mean']:.4f} Std: {stats['std']:.4f}")
    
    def create_visualizations(self, output_dir: str = "big_experiment_results"):
        """Create visualization plots"""
        successful_df = self.df[self.df['evaluator_success'] == True]
        
        if len(successful_df) == 0:
            print("No successful experiments to visualize")
            return
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Model performance comparison
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: MAP comparison
        plt.subplot(2, 2, 1)
        sns.boxplot(data=successful_df, x='vectoriser_model', y='map_mean')
        plt.title('MAP Performance by Model')
        plt.xticks(rotation=45)
        
        # Subplot 2: MRR comparison  
        plt.subplot(2, 2, 2)
        sns.boxplot(data=successful_df, x='vectoriser_model', y='mrr_mean')
        plt.title('MRR Performance by Model')
        plt.xticks(rotation=45)
        
        # Subplot 3: Strategy comparison
        plt.subplot(2, 2, 3)
        sns.boxplot(data=successful_df, x='module_vector_strategy', y='map_mean')
        plt.title('MAP Performance by Strategy')
        plt.xticks(rotation=45)
        
        # Subplot 4: Boolean parameters heatmap
        plt.subplot(2, 2, 4)
        bool_params = ['preprocess_test_task', 'exclude_test_tasks_from_model', 'normalize_vectors']
        bool_df = successful_df[bool_params + ['map_mean']].groupby(bool_params)['map_mean'].mean().reset_index()
        
        if len(bool_df) > 1:
            pivot_df = bool_df.pivot_table(
                values='map_mean', 
                index='normalize_vectors', 
                columns='exclude_test_tasks_from_model'
            )
            sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap='viridis')
            plt.title('MAP: Normalize vs Train/Test Split')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Performance distribution plot
        plt.figure(figsize=(15, 5))
        
        metrics = ['map_mean', 'mrr_mean', 'recall_5_mean', 'recall_10_mean']
        for i, metric in enumerate(metrics, 1):
            if metric in successful_df.columns:
                plt.subplot(1, 4, i)
                successful_df[metric].hist(bins=20, alpha=0.7)
                plt.title(f'{metric.replace("_", " ").title()} Distribution')
                plt.xlabel(metric.replace("_", " ").title())
                plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Parameter correlation heatmap
        plt.figure(figsize=(10, 8))
        
        # Create numeric encoding for categorical variables
        plot_df = successful_df.copy()
        categorical_mapping = {}
        
        for col in ['vectoriser_model', 'module_vector_strategy', 'distance_metrics']:
            if col in plot_df.columns:
                plot_df[f'{col}_encoded'] = pd.Categorical(plot_df[col]).codes
                categorical_mapping[col] = dict(enumerate(plot_df[col].unique()))
        
        # Select numeric columns for correlation
        numeric_cols = [col for col in plot_df.columns if plot_df[col].dtype in ['int64', 'float64', 'bool']]
        numeric_cols = [col for col in numeric_cols if 'duration' not in col and 'success' not in col]
        
        if len(numeric_cols) > 2:
            corr_matrix = plot_df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5)
            plt.title('Parameter Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Model performance over time (experiment order)
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(successful_df['experiment_id'].str.extract('(\d+)').astype(int), 
                   successful_df['map_mean'], alpha=0.6)
        plt.xlabel('Experiment Number')
        plt.ylabel('MAP Mean')
        plt.title('MAP Performance Over Experiment Sequence')
        
        plt.subplot(1, 2, 2)
        plt.scatter(successful_df['total_duration_seconds']/60, successful_df['map_mean'], 
                   alpha=0.6, c=successful_df['embeddings_reused'].map({True: 'green', False: 'red'}))
        plt.xlabel('Duration (minutes)')
        plt.ylabel('MAP Mean')
        plt.title('Performance vs Duration (Green=Reused, Red=New Embeddings)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_trends.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}/")
        print("Created plots:")
        print("  - model_comparison.png: Model and strategy comparisons")
        print("  - performance_distributions.png: Metric distributions")
        print("  - correlation_matrix.png: Parameter correlations")
        print("  - performance_trends.png: Performance over time and duration")
    
    def export_best_configs(self, metric: str = 'map_mean', top_n: int = 5, 
                           output_file: str = None):
        """Export best configurations to a separate CSV"""
        successful_df = self.df[self.df['evaluator_success'] == True]
        
        if len(successful_df) == 0:
            return
        
        if output_file is None:
            output_file = f"big_experiment_results/best_configs_{metric}.csv"
        
        top_configs = successful_df.nlargest(top_n, metric)
        
        # Select key columns for export
        export_columns = [
            'experiment_id', 'vectoriser_model', 'module_vector_strategy',
            'preprocess_test_task', 'exclude_test_tasks_from_model', 'normalize_vectors',
            'distance_metrics', 'map_mean', 'map_std', 'mrr_mean', 'mrr_std',
            'recall_5_mean', 'recall_10_mean', 'recall_20_mean', 'recall_50_mean',
            'total_duration_seconds', 'embeddings_reused'
        ]
        
        export_columns = [col for col in export_columns if col in top_configs.columns]
        
        top_configs[export_columns].to_csv(output_file, index=False)
        print(f"Top {top_n} configurations by {metric} exported to {output_file}")
    
    def interactive_analysis(self):
        """Interactive analysis menu"""
        while True:
            print(f"\n" + "="*50)
            print("BIG EXPERIMENT ANALYZER - INTERACTIVE MODE")
            print("="*50)
            print("1. Summary statistics")
            print("2. Best configurations (MAP)")
            print("3. Best configurations (MRR)")
            print("4. Model comparison")
            print("5. Parameter impact analysis")
            print("6. Create visualizations")
            print("7. Export best configs")
            print("8. Custom metric analysis")
            print("0. Exit")
            
            choice = input("\nEnter your choice (0-8): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self.print_summary_statistics()
            elif choice == '2':
                self.analyze_best_configurations('map_mean', 10)
            elif choice == '3':
                self.analyze_best_configurations('mrr_mean', 10)
            elif choice == '4':
                metric = input("Enter metric (default: map_mean): ").strip() or 'map_mean'
                self.compare_models(metric)
            elif choice == '5':
                metric = input("Enter metric (default: map_mean): ").strip() or 'map_mean'
                self.analyze_parameter_impact(metric)
            elif choice == '6':
                self.create_visualizations()
            elif choice == '7':
                metric = input("Enter metric (default: map_mean): ").strip() or 'map_mean'
                top_n = int(input("Enter number of configs (default: 5): ").strip() or '5')
                self.export_best_configs(metric, top_n)
            elif choice == '8':
                metric = input("Enter metric name: ").strip()
                if metric:
                    self.analyze_best_configurations(metric, 10)
                    self.compare_models(metric)
            else:
                print("Invalid choice!")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze big experiment results')
    parser.add_argument('--csv', default='big_experiment_results/experiment_summary.csv',
                       help='Path to experiment results CSV')
    parser.add_argument('--summary', action='store_true',
                       help='Print summary statistics and exit')
    parser.add_argument('--best', type=int, default=0,
                       help='Show top N best configurations and exit')
    parser.add_argument('--metric', default='map_mean',
                       help='Metric to use for analysis (default: map_mean)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations and exit')
    parser.add_argument('--export', action='store_true',
                       help='Export best configurations and exit')
    
    args = parser.parse_args()
    
    analyzer = ExperimentAnalyzer(args.csv)
    
    if args.summary:
        analyzer.print_summary_statistics()
    elif args.best > 0:
        analyzer.analyze_best_configurations(args.metric, args.best)
    elif args.visualize:
        analyzer.create_visualizations()
    elif args.export:
        analyzer.export_best_configs(args.metric)
    else:
        # Interactive mode
        analyzer.interactive_analysis()

if __name__ == "__main__":
    main()