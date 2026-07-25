"""
Resume ETL Pipeline from checkpoint
Checks what collections already exist and only processes missing ones
"""

from qdrant_client import QdrantClient
import config
import argparse
import sys

def check_existing_collections(client):
    """Get list of existing RAG experiment collections"""
    try:
        collections = client.get_collections().collections
        existing = [c.name for c in collections if c.name.startswith(config.COLLECTION_PREFIX)]
        return set(existing)
    except Exception as e:
        print(f"ERROR: Cannot connect to Qdrant: {e}")
        print("Make sure Qdrant is running: podman-compose up -d")
        sys.exit(1)

def get_missing_variants(split_strategy='recent'):
    """Determine which source/target/window combinations are missing"""

    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    existing = check_existing_collections(client)

    print("="*80)
    print("EXISTING COLLECTIONS:")
    print("="*80)
    for name in sorted(existing):
        print(f"  ✓ {name}")
    print(f"\nTotal: {len(existing)} collections")
    print()

    # Build expected collections
    all_combinations = []
    for source in config.SOURCE_VARIANTS.keys():
        for target in config.TARGET_VARIANTS.keys():
            for window in config.WINDOW_VARIANTS.keys():
                collection_name = f"{config.COLLECTION_PREFIX}_{source}_{target}_{window}_{split_strategy}"
                all_combinations.append({
                    'name': collection_name,
                    'source': source,
                    'target': target,
                    'window': window
                })

    # Find missing
    missing = [c for c in all_combinations if c['name'] not in existing]

    print("="*80)
    print("MISSING COLLECTIONS:")
    print("="*80)
    if missing:
        for c in missing:
            print(f"  ✗ {c['name']}")
        print(f"\nTotal missing: {len(missing)}")
    else:
        print("  None - All collections complete!")
    print()

    return missing, len(all_combinations)

def main():
    parser = argparse.ArgumentParser(description='Resume ETL Pipeline from checkpoint')
    parser.add_argument(
        '--split_strategy',
        type=str,
        default='recent',
        choices=['recent', 'modn'],
        help='Split strategy used (must match original)'
    )
    parser.add_argument(
        '--show-only',
        action='store_true',
        help='Only show status, do not resume'
    )

    args = parser.parse_args()

    missing, total = get_missing_variants(args.split_strategy)

    print("="*80)
    print("SUMMARY:")
    print("="*80)
    print(f"Expected collections: {total}")
    print(f"Completed: {total - len(missing)}")
    print(f"Missing: {len(missing)}")
    print(f"Progress: {(total - len(missing)) / total * 100:.1f}%")
    print("="*80)
    print()

    if args.show_only or not missing:
        return

    # Group missing by source and window
    sources_needed = set(c['source'] for c in missing)
    windows_needed = set(c['window'] for c in missing)

    print("TO RESUME, RUN:")
    print("="*80)
    print(f"python etl_pipeline.py --split_strategy {args.split_strategy} \\")
    print(f"    --sources {' '.join(sorted(sources_needed))} \\")
    print(f"    --windows {' '.join(sorted(windows_needed))}")
    print()
    print("This will only process the missing collections.")
    print("="*80)

if __name__ == '__main__':
    main()
