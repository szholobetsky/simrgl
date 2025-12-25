#!/usr/bin/env python3
"""
Switch ragmcp config to use test collections (w1000)
For quick testing without waiting for full ETL
"""

import os
import sys

def update_config(use_test_mode=True):
    """Update config.py to use test or production collections"""

    config_path = os.path.join(os.path.dirname(__file__), 'config.py')

    # Read current config
    with open(config_path, 'r') as f:
        lines = f.readlines()

    # Update collection names
    new_lines = []
    for line in lines:
        if line.startswith('COLLECTION_MODULE ='):
            if use_test_mode:
                new_lines.append("COLLECTION_MODULE = 'rag_exp_desc_module_w1000_modn_bge-small'  # TEST MODE - Last 1000 tasks\n")
            else:
                new_lines.append("COLLECTION_MODULE = 'rag_exp_desc_module_all_modn_bge-small'  # Main collection for module search\n")
        elif line.startswith('COLLECTION_FILE ='):
            if use_test_mode:
                new_lines.append("COLLECTION_FILE = 'rag_exp_desc_file_w1000_modn_bge-small'      # TEST MODE - Last 1000 tasks\n")
            else:
                new_lines.append("COLLECTION_FILE = 'rag_exp_desc_file_all_modn_bge-small'      # Alternative file-level collection\n")
        else:
            new_lines.append(line)

    # Write updated config
    with open(config_path, 'w') as f:
        f.writelines(new_lines)

    mode = "TEST (w1000)" if use_test_mode else "PRODUCTION (all)"
    print(f"✓ Config updated to use {mode} collections")
    print(f"  COLLECTION_MODULE: rag_exp_desc_module_{'w1000' if use_test_mode else 'all'}_modn_bge-small")
    print(f"  COLLECTION_FILE: rag_exp_desc_file_{'w1000' if use_test_mode else 'all'}_modn_bge-small")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Switch between test and production collections')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['test', 'production'],
        default='test',
        help='Which collections to use'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Collection Switcher")
    print("=" * 60)
    print()

    use_test = (args.mode == 'test')
    update_config(use_test)

    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)
    print()
    print("Next steps:")
    if use_test:
        print("1. Make sure test collections exist:")
        print("   cd ../exp3")
        print("   run_etl_test_qdrant.bat  (or run_etl_test_postgres.bat)")
        print()
        print("2. Launch Gradio UI:")
        print("   python gradio_ui.py")
        print()
        print("3. Test the RAG system with fast collections!")
    else:
        print("1. Make sure production collections exist:")
        print("   cd ../exp3")
        print("   run_etl_practical.bat  (or run_etl_postgres.bat)")
        print()
        print("2. Launch Gradio UI:")
        print("   python gradio_ui.py")
        print()
        print("3. Use full RAG system with all tasks!")
    print("=" * 60)
