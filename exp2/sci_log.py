import os
import shutil
import json
import csv
import time
from datetime import datetime
from typing import Dict, Any, Optional


class SciLog:
    def __init__(self):
        self.log_dir = "log"
        self.run_number: Optional[int] = None
        self.start_time: Optional[float] = None
        self.start_datetime: Optional[datetime] = None
        self.key_values: Dict[str, Any] = {}
        self.original_file: Optional[str] = None
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
    
    def _get_next_run_number(self) -> int:
        """Get the next available run number by checking existing files."""
        existing_files = os.listdir(self.log_dir)
        run_numbers = []
        
        for filename in existing_files:
            if filename.startswith("script_") and filename.endswith(".py"):
                try:
                    num_str = filename[7:-3]  # Remove "script_" and ".py"
                    run_numbers.append(int(num_str))
                except ValueError:
                    continue
        
        return max(run_numbers, default=0) + 1
    
    def start(self, file_path: str) -> None:
        """
        Start logging experiment. Copy the current script to log folder.
        
        Args:
            file_path: Path to the current script file (use __file__)
        """
        # Record start time
        self.start_time = time.time()
        self.start_datetime = datetime.now()
        
        # Get next run number
        self.run_number = self._get_next_run_number()
        
        # Store original file path
        self.original_file = file_path
        
        # Copy script to log folder
        script_name = f"script_{self.run_number:03d}.py"
        script_path = os.path.join(self.log_dir, script_name)
        shutil.copy2(file_path, script_path)
        
        # Initialize key_values with automatic timing keys
        self.key_values = {
            'start': self.start_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'Fstart': self.start_time
        }
        
        print(f"Experiment started - Run #{self.run_number}")
        print(f"Script copied to: {script_path}")
    
    def key(self, key: str, value: Any) -> None:
        """
        Add a key-value pair to the experiment log.
        
        Args:
            key: The key name
            value: The value to store
        """
        if self.run_number is None:
            raise RuntimeError("Must call start() before adding keys")
        
        self.key_values[key] = value
        print(f"Logged: {key} = {value}")
    
    def stop(self, format_type: str = 'csv') -> None:
        """
        Stop logging and save the data.
        
        Args:
            format_type: Output format - 'csv' or 'json'
        """
        if self.run_number is None:
            raise RuntimeError("Must call start() before stop()")
        
        if self.start_time is None:
            raise RuntimeError("Start time not recorded")
        
        # Record stop time
        stop_time = time.time()
        stop_datetime = datetime.now()
        
        # Add automatic timing keys
        self.key_values.update({
            'stop': stop_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'Fstop': stop_time,
            'diff': stop_time - self.start_time
        })
        
        # Save data based on format
        if format_type.lower() == 'csv':
            self._save_csv()
        elif format_type.lower() == 'json':
            self._save_json()
        else:
            raise ValueError("Format must be 'csv' or 'json'")
        
        print(f"Experiment stopped - Duration: {self.key_values['diff']:.3f} seconds")
        
        # Reset for next experiment
        self._reset()
    
    def _save_csv(self) -> None:
        """Save key-values to CSV file."""
        filename = f"key_{self.run_number:03d}.csv"
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['key', 'value'])  # Header
            for key, value in self.key_values.items():
                writer.writerow([key, value])
        
        print(f"Data saved to: {filepath}")
    
    def _save_json(self) -> None:
        """Save key-values to JSON file."""
        filename = f"key_{self.run_number:03d}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as jsonfile:
            json.dump(self.key_values, jsonfile, indent=2, default=str)
        
        print(f"Data saved to: {filepath}")
    
    def _reset(self) -> None:
        """Reset the logger state."""
        self.run_number = None
        self.start_time = None
        self.start_datetime = None
        self.key_values = {}
        self.original_file = None


# Create singleton instance
sci_log = SciLog()


# Example usage and test
if __name__ == "__main__":
    # Example of how to use the framework
    print("=== SciLog Framework Example ===")
    
    # Start experiment
    sci_log.start(__file__)
    
    # Simulate some experimental work
    time.sleep(0.1)  # Simulate processing time
    
    # Log some experimental parameters
    sci_log.key('temperature', 25.5)
    sci_log.key('pressure', 1013.25)
    sci_log.key('sample_id', 'S001')
    sci_log.key('ph_level', 7.2)
    sci_log.key('concentration', 0.05)
    
    # Simulate more work
    time.sleep(0.1)
    
    # Log results
    sci_log.key('result', 'success')
    sci_log.key('yield_percentage', 85.3)
    
    # Stop and save as CSV
    sci_log.stop('csv')
    
    print("\n" + "="*50)
    print("Example with JSON format:")
    
    # Second experiment example
    sci_log.start(__file__)
    sci_log.key('experiment_type', 'control')
    sci_log.key('iterations', 100)
    time.sleep(0.05)
    sci_log.stop('json')