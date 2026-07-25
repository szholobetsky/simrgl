"""
Backup and Restore Qdrant Collections
Exports all vectors and metadata to JSON files for portability
"""

import os
import json
import argparse
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm

import config


def get_client():
    """Get Qdrant client"""
    return QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)


def backup_collections(output_dir: str):
    """
    Backup all collections to JSON files

    Args:
        output_dir: Directory to save backup files
    """
    client = get_client()

    # Get all collections
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    print(f"Found {len(collection_names)} collections to backup")

    os.makedirs(output_dir, exist_ok=True)

    for coll_name in collection_names:
        print(f"\nBacking up: {coll_name}")

        # Get collection info
        coll_info = client.get_collection(coll_name)
        vector_size = coll_info.config.params.vectors.size
        distance = coll_info.config.params.vectors.distance.value

        # Get all points (scroll through all)
        all_points = []
        offset = None

        while True:
            results, offset = client.scroll(
                collection_name=coll_name,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )

            for point in results:
                all_points.append({
                    'id': point.id,
                    'vector': point.vector,
                    'payload': point.payload
                })

            if offset is None:
                break

        print(f"  Exported {len(all_points)} points")

        # Save to file
        backup_data = {
            'collection_name': coll_name,
            'vector_size': vector_size,
            'distance': distance,
            'points': all_points
        }

        output_file = os.path.join(output_dir, f"{coll_name}.json")
        with open(output_file, 'w') as f:
            json.dump(backup_data, f)

        print(f"  Saved to: {output_file}")

    # Save manifest
    manifest = {
        'collections': collection_names,
        'total_collections': len(collection_names)
    }
    manifest_file = os.path.join(output_dir, 'manifest.json')
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nBackup complete! {len(collection_names)} collections saved to {output_dir}")


def restore_collections(input_dir: str):
    """
    Restore collections from JSON backup files

    Args:
        input_dir: Directory containing backup files
    """
    client = get_client()

    # Load manifest
    manifest_file = os.path.join(input_dir, 'manifest.json')
    if not os.path.exists(manifest_file):
        print("ERROR: manifest.json not found in backup directory")
        return False

    with open(manifest_file, 'r') as f:
        manifest = json.load(f)

    collection_names = manifest['collections']
    print(f"Found {len(collection_names)} collections to restore")

    for coll_name in collection_names:
        backup_file = os.path.join(input_dir, f"{coll_name}.json")

        if not os.path.exists(backup_file):
            print(f"WARNING: Backup file not found for {coll_name}, skipping")
            continue

        print(f"\nRestoring: {coll_name}")

        # Load backup data
        with open(backup_file, 'r') as f:
            backup_data = json.load(f)

        vector_size = backup_data['vector_size']
        distance = backup_data['distance']
        points = backup_data['points']

        # Delete existing collection if exists
        try:
            client.delete_collection(coll_name)
            print(f"  Deleted existing collection")
        except:
            pass

        # Create collection
        distance_enum = models.Distance[distance.upper()]
        client.create_collection(
            collection_name=coll_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=distance_enum
            )
        )
        print(f"  Created collection (vector_size={vector_size}, distance={distance})")

        # Insert points in batches
        batch_size = 100
        for i in tqdm(range(0, len(points), batch_size), desc="  Inserting"):
            batch = points[i:i + batch_size]
            point_structs = [
                models.PointStruct(
                    id=p['id'],
                    vector=p['vector'],
                    payload=p['payload']
                )
                for p in batch
            ]
            client.upsert(
                collection_name=coll_name,
                points=point_structs,
                wait=True
            )

        print(f"  Restored {len(points)} points")

    print(f"\nRestore complete! {len(collection_names)} collections restored")
    return True


def main():
    parser = argparse.ArgumentParser(description='Backup/Restore Qdrant Collections')
    parser.add_argument(
        '--action',
        type=str,
        required=True,
        choices=['backup', 'restore'],
        help='Action to perform'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='qdrant_snapshots',
        help='Output directory for backup'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='qdrant_snapshots',
        help='Input directory for restore'
    )

    args = parser.parse_args()

    if args.action == 'backup':
        backup_collections(args.output)
    elif args.action == 'restore':
        restore_collections(args.input)


if __name__ == '__main__':
    main()
