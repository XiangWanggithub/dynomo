#!/usr/bin/env python3
"""
Example script to prepare and validate custom dataset structure.

This script helps you:
1. Check if your dataset has the correct structure
2. Validate that all required files are present
3. Generate a summary of your dataset
"""

import os
import glob
from pathlib import Path
from natsort import natsorted
import numpy as np


def check_directory_structure(basedir, sequence):
    """
    Check if dataset directory structure is correct.

    Args:
        basedir: Base directory (e.g., "data/custom")
        sequence: Sequence name (e.g., "my_sequence")

    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }

    sequence_dir = os.path.join(basedir, sequence)

    # Check if sequence directory exists
    if not os.path.exists(sequence_dir):
        results['valid'] = False
        results['errors'].append(f"Sequence directory not found: {sequence_dir}")
        return results

    # Check RGB directory
    rgb_dir = os.path.join(sequence_dir, 'rgb')
    if not os.path.exists(rgb_dir):
        results['valid'] = False
        results['errors'].append(f"RGB directory not found: {rgb_dir}")
        return results

    # Get RGB files
    rgb_extensions = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']
    rgb_files = []
    for ext in rgb_extensions:
        rgb_files.extend(glob.glob(f"{rgb_dir}/{ext}"))
    rgb_files = natsorted(rgb_files)

    if len(rgb_files) == 0:
        results['valid'] = False
        results['errors'].append(f"No RGB images found in {rgb_dir}")
        return results

    results['info']['num_rgb_files'] = len(rgb_files)
    results['info']['rgb_files'] = rgb_files

    # Check SAM directory
    sam_dir = os.path.join(sequence_dir, 'sam')
    if not os.path.exists(sam_dir):
        results['valid'] = False
        results['errors'].append(f"SAM directory not found: {sam_dir}")
        return results

    # Get SAM files
    sam_extensions = ['*.png', '*.jpg', '*.npy']
    sam_files = []
    for ext in sam_extensions:
        sam_files.extend(glob.glob(f"{sam_dir}/{ext}"))
    sam_files = natsorted(sam_files)

    if len(sam_files) == 0:
        results['valid'] = False
        results['errors'].append(f"No SAM masks found in {sam_dir}")
        return results

    if len(sam_files) != len(rgb_files):
        results['warnings'].append(
            f"Number of SAM masks ({len(sam_files)}) doesn't match "
            f"number of RGB images ({len(rgb_files)})"
        )

    results['info']['num_sam_files'] = len(sam_files)

    # Check depth directory (optional)
    depth_dir = os.path.join(sequence_dir, 'depth')
    if os.path.exists(depth_dir):
        depth_extensions = ['*.npy', '*.png', '*.exr']
        depth_files = []
        for ext in depth_extensions:
            depth_files.extend(glob.glob(f"{depth_dir}/{ext}"))
        depth_files = natsorted(depth_files)

        results['info']['num_depth_files'] = len(depth_files)
        results['info']['has_depth'] = True

        if len(depth_files) != len(rgb_files):
            results['warnings'].append(
                f"Number of depth maps ({len(depth_files)}) doesn't match "
                f"number of RGB images ({len(rgb_files)})"
            )
    else:
        results['info']['has_depth'] = False
        results['warnings'].append(
            "Depth directory not found. You'll need to use online depth estimation."
        )

    # Check embeddings directory (optional)
    emb_dir = os.path.join(sequence_dir, 'embeddings')
    if os.path.exists(emb_dir):
        emb_files = natsorted(glob.glob(f"{emb_dir}/*.npy"))
        results['info']['num_embedding_files'] = len(emb_files)
        results['info']['has_embeddings'] = True

        if len(emb_files) != len(rgb_files):
            results['warnings'].append(
                f"Number of embeddings ({len(emb_files)}) doesn't match "
                f"number of RGB images ({len(rgb_files)})"
            )
    else:
        results['info']['has_embeddings'] = False
        results['warnings'].append(
            "Embeddings directory not found. You'll need to use online embedding extraction."
        )

    # Check poses file (optional)
    pose_files = ['poses.npy', 'poses.txt']
    for pose_file in pose_files:
        pose_path = os.path.join(sequence_dir, pose_file)
        if os.path.exists(pose_path):
            results['info']['has_poses'] = True
            results['info']['pose_file'] = pose_file
            break
    else:
        results['info']['has_poses'] = False
        results['warnings'].append(
            "Pose file not found. Identity poses will be used."
        )

    return results


def print_validation_results(results, basedir, sequence):
    """Print validation results in a readable format."""
    print("\n" + "="*70)
    print(f"Dataset Validation Results")
    print(f"Sequence: {sequence}")
    print(f"Directory: {os.path.join(basedir, sequence)}")
    print("="*70)

    if results['valid']:
        print("\n✓ Dataset structure is VALID")
    else:
        print("\n✗ Dataset structure is INVALID")

    # Print errors
    if results['errors']:
        print("\nErrors:")
        for error in results['errors']:
            print(f"  ✗ {error}")

    # Print warnings
    if results['warnings']:
        print("\nWarnings:")
        for warning in results['warnings']:
            print(f"  ⚠ {warning}")

    # Print info
    if results['info']:
        print("\nDataset Information:")
        print(f"  - RGB images: {results['info'].get('num_rgb_files', 0)}")
        print(f"  - SAM masks: {results['info'].get('num_sam_files', 0)}")

        if results['info'].get('has_depth'):
            print(f"  - Depth maps: {results['info'].get('num_depth_files', 0)}")
        else:
            print(f"  - Depth maps: Not found (will use online estimation)")

        if results['info'].get('has_embeddings'):
            print(f"  - Embeddings: {results['info'].get('num_embedding_files', 0)}")
        else:
            print(f"  - Embeddings: Not found (will use online extraction)")

        if results['info'].get('has_poses'):
            print(f"  - Poses: Found ({results['info'].get('pose_file')})")
        else:
            print(f"  - Poses: Not found (will use identity poses)")

    print("\n" + "="*70)

    # Print recommendations
    if results['valid']:
        print("\nRecommendations:")
        print("\n1. Update camera parameters in: configs/data/custom.yaml")
        print("2. Update config file: configs/custom/dynomo_custom.py")
        print("   - Set scene_name = '{}'".format(sequence))
        print("   - Set basedir = '{}'".format(basedir))

        print("\n3. Train the model:")
        print("   python scripts/train_custom.py \\")
        print("       --config configs/custom/dynomo_custom.py \\")
        print("       --sequence {} \\".format(sequence))
        print("       --gpus 0 \\")

        if not results['info'].get('has_depth'):
            print("       --online_depth DepthAnythingV2-vitl \\")

        if not results['info'].get('has_embeddings'):
            print("       --online_emb dinov2_vits14")
        else:
            print("")

    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate custom dataset structure for DynOMo"
    )
    parser.add_argument(
        "--basedir",
        type=str,
        required=True,
        help="Base directory (e.g., data/custom)"
    )
    parser.add_argument(
        "--sequence",
        type=str,
        required=True,
        help="Sequence name"
    )

    args = parser.parse_args()

    # Validate dataset
    results = check_directory_structure(args.basedir, args.sequence)

    # Print results
    print_validation_results(results, args.basedir, args.sequence)

    # Return exit code
    return 0 if results['valid'] else 1


if __name__ == "__main__":
    exit(main())
