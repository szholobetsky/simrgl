#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Big Experiment
Simple script to execute the comprehensive parameter sweep and analyze results
"""

import sys
import os
import subprocess
from datetime import datetime

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=False, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"💥 {description} failed with exception: {e}")
        return False

def main():
    """Main execution function"""
    start_time = datetime.now()
    print(f"🚀 Starting Big Experiment Suite at {start_time}")
    
    # Check if required files exist
    required_files = [
        'big_experiment.py',
        'experiment_analyzer.py', 
        'properties.py',
        'main.py',
        'vectoriser.py',
        'statistical_evaluator.py'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        sys.exit(1)
    
    print(f"✅ All required files found")
    
    # Run the big experiment
    success = run_command(
        "python big_experiment.py",
        "Big Parameter Sweep Experiment"
    )
    
    if not success:
        print("\n❌ Big experiment failed. Check the error messages above.")
        sys.exit(1)
    
    # Run analysis
    analysis_success = run_command(
        "python experiment_analyzer.py --summary",
        "Experiment Results Summary"
    )
    
    if analysis_success:
        run_command(
            "python experiment_analyzer.py --best 10",
            "Top 10 Best Configurations"
        )
        
        run_command(
            "python experiment_analyzer.py --visualize",
            "Results Visualization"
        )
        
        run_command(
            "python experiment_analyzer.py --export",
            "Export Best Configurations"
        )
    
    # Print final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n🎉 Big Experiment Suite completed!")
    print(f"Started: {start_time}")
    print(f"Finished: {end_time}")
    print(f"Total duration: {duration}")
    print(f"\nResults available in:")
    print(f"  📊 big_experiment_results/experiment_summary.csv")
    print(f"  📈 big_experiment_results/*.png (visualizations)")
    print(f"  📋 big_experiment_results/best_configs_*.csv")
    print(f"  📁 big_experiment_results/statistical_outputs/")
    print(f"  📝 big_experiment_results/sci_logs/")
    
    print(f"\nTo run interactive analysis:")
    print(f"  python experiment_analyzer.py")

if __name__ == "__main__":
    main()