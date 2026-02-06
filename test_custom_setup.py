#!/usr/bin/env python3
"""
Quick test script to verify custom dataset setup.

This script tests:
1. CustomDataset can be imported
2. Configuration files are valid
3. Basic functionality works
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.getcwd())


def test_imports():
    """Test if all necessary modules can be imported."""
    print("Testing imports...")

    try:
        from src.datasets.datasets import CustomDataset
        print("  ✓ CustomDataset imported successfully")
    except Exception as e:
        print(f"  ✗ Failed to import CustomDataset: {e}")
        return False

    try:
        from src.model.dynomo import DynOMo
        print("  ✓ DynOMo imported successfully")
    except Exception as e:
        print(f"  ✗ Failed to import DynOMo: {e}")
        return False

    try:
        from src.utils.get_data import get_data, get_dataset
        print("  ✓ Data utilities imported successfully")
    except Exception as e:
        print(f"  ✗ Failed to import data utilities: {e}")
        return False

    return True


def test_config_files():
    """Test if configuration files exist and are valid."""
    print("\nTesting configuration files...")

    # Check config file
    config_file = "configs/custom/dynomo_custom.py"
    if not os.path.exists(config_file):
        print(f"  ✗ Config file not found: {config_file}")
        return False
    print(f"  ✓ Config file exists: {config_file}")

    # Try to load config
    try:
        from importlib.machinery import SourceFileLoader
        config_module = SourceFileLoader(
            os.path.basename(config_file), config_file
        ).load_module()
        config = config_module.config
        print(f"  ✓ Config loaded successfully")

        # Check required keys
        required_keys = ['data', 'tracking_obj', 'tracking_cam', 'viz']
        for key in required_keys:
            if key not in config:
                print(f"  ✗ Missing required config key: {key}")
                return False
        print(f"  ✓ All required config keys present")

    except Exception as e:
        print(f"  ✗ Failed to load config: {e}")
        return False

    # Check data config file
    data_config_file = "configs/data/custom.yaml"
    if not os.path.exists(data_config_file):
        print(f"  ✗ Data config file not found: {data_config_file}")
        return False
    print(f"  ✓ Data config file exists: {data_config_file}")

    return True


def test_scripts():
    """Test if scripts exist and are executable."""
    print("\nTesting scripts...")

    scripts = [
        "scripts/train_custom.py",
        "scripts/inference_custom.py",
        "examples/prepare_custom_data.py"
    ]

    for script in scripts:
        if not os.path.exists(script):
            print(f"  ✗ Script not found: {script}")
            return False
        print(f"  ✓ Script exists: {script}")

    return True


def test_dataset_class():
    """Test CustomDataset class instantiation."""
    print("\nTesting CustomDataset class...")

    # This test will fail if no actual data exists, but tests the class definition
    try:
        from src.datasets.datasets import CustomDataset, load_dataset_config

        # Load config
        config_dict = load_dataset_config("configs/data/custom.yaml")
        print(f"  ✓ Config loaded successfully")

        # Check if required methods exist
        required_methods = [
            'get_filepaths',
            'load_poses',
            '_load_bg',
            '_load_instseg',
            'read_embedding_from_file'
        ]

        for method in required_methods:
            if not hasattr(CustomDataset, method):
                print(f"  ✗ Missing required method: {method}")
                return False

        print(f"  ✓ All required methods present")

    except Exception as e:
        print(f"  ✗ Failed to test CustomDataset: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("="*70)
    print("Custom Dataset Setup Test")
    print("="*70)

    tests = [
        ("Imports", test_imports),
        ("Configuration Files", test_config_files),
        ("Scripts", test_scripts),
        ("CustomDataset Class", test_dataset_class),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))

    # Print summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)

    all_passed = True
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False

    print("="*70)

    if all_passed:
        print("\n🎉 All tests passed! Your setup is ready to use.")
        print("\nNext steps:")
        print("1. Prepare your data in the required format")
        print("2. Validate your data: python examples/prepare_custom_data.py --basedir data/custom --sequence your_sequence")
        print("3. Update camera parameters in: configs/data/custom.yaml")
        print("4. Update config file: configs/custom/dynomo_custom.py")
        print("5. Start training: python scripts/train_custom.py --config configs/custom/dynomo_custom.py --sequence your_sequence --gpus 0")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
