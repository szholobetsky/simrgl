#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Properties File Updater
Updates properties.py file with new values for experiments
"""

import re
import os
from typing import Dict, Any

class PropertiesUpdater:
    def __init__(self, properties_file: str = "properties.py"):
        self.properties_file = properties_file
        self.backup_file = f"{properties_file}.backup"
    
    def backup_properties(self):
        """Create a backup of the original properties file"""
        if os.path.exists(self.properties_file):
            with open(self.properties_file, 'r', encoding='utf-8') as src:
                with open(self.backup_file, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
    
    def restore_properties(self):
        """Restore properties from backup"""
        if os.path.exists(self.backup_file):
            with open(self.backup_file, 'r', encoding='utf-8') as src:
                with open(self.properties_file, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
    
    def update_properties(self, updates: Dict[str, Any]):
        """Update properties file with new values"""
        if not os.path.exists(self.properties_file):
            raise FileNotFoundError(f"Properties file {self.properties_file} not found")
        
        # Read current content
        with open(self.properties_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update each property
        for key, value in updates.items():
            content = self._update_single_property(content, key, value)
        
        # Write updated content
        with open(self.properties_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _update_single_property(self, content: str, key: str, value: Any) -> str:
        """Update a single property in the content"""
        # Convert value to string representation
        if isinstance(value, str):
            value_str = f"'{value}'"
        elif isinstance(value, bool):
            value_str = str(value)
        elif isinstance(value, (int, float)):
            value_str = str(value)
        elif isinstance(value, list):
            # Handle list values
            if all(isinstance(item, str) for item in value):
                value_str = "[" + ", ".join([f"'{item}'" for item in value]) + "]"
            else:
                value_str = str(value)
        elif value is None:
            value_str = "None"
        else:
            value_str = str(value)
        
        # Pattern to match the property assignment
        # Matches: PROPERTY_NAME = value (with optional comments)
        pattern = rf'^({key}\s*=\s*)([^#\n]*)(.*?)$'
        
        # Find and replace the property
        lines = content.split('\n')
        updated_lines = []
        found = False
        
        for line in lines:
            match = re.match(pattern, line)
            if match:
                # Keep the property name and assignment operator
                prefix = match.group(1)
                # Keep any comments at the end of the line
                suffix = match.group(3).rstrip() if match.group(3).strip() else ""
                # Create new line with updated value
                new_line = f"{prefix}{value_str}{suffix}"
                updated_lines.append(new_line)
                found = True
            else:
                updated_lines.append(line)
        
        if not found:
            # Property not found, add it at the end
            updated_lines.append(f"{key} = {value_str}")
        
        return '\n'.join(updated_lines)
    
    def get_current_value(self, key: str) -> str:
        """Get current value of a property (as string)"""
        if not os.path.exists(self.properties_file):
            return ""
        
        with open(self.properties_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        pattern = rf'^{key}\s*=\s*([^#\n]*)'
        
        for line in content.split('\n'):
            match = re.match(pattern, line)
            if match:
                return match.group(1).strip()
        
        return ""

# Convenience functions for the experiment runner
def update_experiment_properties(model_config: Dict[str, Any], strategy: str, 
                                clear_embeddings: bool, fixed_params: Dict[str, Any]):
    """Update properties file for a specific experiment"""
    updater = PropertiesUpdater()
    
    # Combine all updates
    updates = {}
    
    # Add fixed parameters
    updates.update(fixed_params)
    
    # Add model-specific parameters (skip description)
    for param, value in model_config.items():
        if param != 'model_description':
            updates[param] = value
    
    # Add strategy and embedding settings
    updates['MODULE_VECTOR_STRATEGY'] = strategy
    updates['CLEAR_EMBEDDINGS'] = clear_embeddings
    
    # Apply updates
    updater.update_properties(updates)

def backup_original_properties():
    """Create backup of original properties"""
    updater = PropertiesUpdater()
    updater.backup_properties()

def restore_original_properties():
    """Restore original properties from backup"""
    updater = PropertiesUpdater()
    updater.restore_properties()

# Example usage
if __name__ == "__main__":
    # Test the updater
    updater = PropertiesUpdater()
    
    # Backup original
    updater.backup_properties()
    
    # Test updates
    test_updates = {
        'VECTORISER_MODEL': 'bert',
        'BERT_MODEL_NAME': 'bert-base-uncased',
        'MODULE_VECTOR_STRATEGY': 'weighted_avg',
        'CLEAR_EMBEDDINGS': True,
        'DISTANCE_METRICS': ['cosine', 'euclidean']
    }
    
    print("Updating properties...")
    updater.update_properties(test_updates)
    
    print("Properties updated. Check properties.py file.")
    print("Run restore_original_properties() to revert changes.")