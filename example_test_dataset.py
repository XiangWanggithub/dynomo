"""
Minimal Example: Testing Dataset Loading

This script demonstrates the simplest way to test if your dataset loads correctly.
Use this to verify your data is structured properly before running full inference.

Usage:
    python example_test_dataset.py
"""

import os
import sys
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.datasets.datasets.custom_test import CustomTestDataset


def main():
    """Minimal example of loading a custom dataset."""

    print("=" * 60)
    print("Custom Dataset Loading Example")
    print("=" * 60)

    # Configuration dictionary with camera parameters
    config_dict = {
        'camera_params': {
            'image_height': 480,
            'image_width': 640,
            'fx': 525.0,  # Focal length x (pixels)
            'fy': 525.0,  # Focal length y (pixels)
            'cx': 320.0,  # Principal point x (pixels)
            'cy': 240.0,  # Principal point y (pixels)
            'png_depth_scale': 1.0,
        },
    }

    # Dataset parameters
    basedir = "/path/to/your/data"  # UPDATE THIS: e.g., "/data/videos"
    sequence = "test_sequence"      # UPDATE THIS: e.g., "my_video"

    print(f"\nLoading dataset from: {basedir}/{sequence}")
    print("Expected structure:")
    print(f"  {basedir}/{sequence}/rgb/")
    print(f"  {basedir}/{sequence}/rgb/0000.png")
    print(f"  {basedir}/{sequence}/rgb/0001.png")
    print("  ...")
    print()

    try:
        # Create dataset
        dataset = CustomTestDataset(
            config_dict=config_dict,
            basedir=basedir,
            sequence=sequence,
            desired_height=480,
            desired_width=640,
            online_depth='DepthAnythingV2',  # Use online depth prediction
            load_embeddings=False,           # Don't load embeddings
            use_dummy_segmentation=True,     # Use dummy segmentation masks
        )

        print(f"\n✓ Dataset loaded successfully!")
        print(f"  Number of frames: {len(dataset)}")

        # Test loading first frame
        print("\nTesting first frame...")
        color, depth, intrinsics, pose, embeddings, bg, instseg = dataset[0]

        print(f"  ✓ RGB shape: {color.shape} (should be [3, H, W])")
        print(f"  ✓ Depth shape: {depth.shape} (should be [1, H, W])")
        print(f"  ✓ Intrinsics shape: {intrinsics.shape} (should be [4, 4])")
        print(f"  ✓ Pose shape: {pose.shape} (should be [4, 4])")
        print(f"  ✓ Background mask shape: {bg.shape} (should be [1, H, W])")
        print(f"  ✓ Instance seg shape: {instseg.shape} (should be [1, H, W])")

        # Check value ranges
        print("\nValue ranges:")
        print(f"  RGB: [{color.min():.3f}, {color.max():.3f}] (should be [0, 1])")
        print(f"  Depth: [{depth.min():.3f}, {depth.max():.3f}]")

        print("\n" + "=" * 60)
        print("SUCCESS! Your dataset is properly configured.")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Review the output above to verify shapes and values")
        print("2. Run full visualization:")
        print(f"   python test_without_gt.py --config configs/custom/test_custom.py --sequence {sequence} --visualize_only")
        print()

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure basedir and sequence are set correctly above")
        print("2. Verify the RGB folder exists:")
        print(f"   ls {basedir}/{sequence}/rgb/")
        print("3. Check that images are named sequentially (0000.png, 0001.png, ...)")
        print()

    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        print("\nFull error details:")
        import traceback
        traceback.print_exc()
        print()


if __name__ == "__main__":
    main()
