#!/usr/bin/env python3
"""
Training/Fine-tuning Script for Custom Dataset Online Tracking

This script performs parameter fine-tuning on custom datasets with RGB, Depth, and SAM segmentation.

Usage:
    python scripts/train_custom.py \
        --config configs/custom/dynomo_custom.py \
        --sequence your_sequence_name \
        --gpus 0

Optional arguments:
    --online_depth DepthAnythingV2-vitl  # Use online depth estimation
    --online_emb dinov2_vits14           # Use online embedding extraction
    --checkpoint path/to/checkpoint.npz  # Resume from checkpoint
"""

import os
import sys
import argparse
from importlib.machinery import SourceFileLoader

# Add parent directory to path
sys.path.append(os.getcwd())

from src.utils.common_utils import seed_everything
from src.model.dynomo import DynOMo
import json


def train_custom_sequence(
    config_file: str,
    sequence: str,
    gpu_id: int = 0,
    online_depth: str = None,
    online_emb: str = None,
    checkpoint_path: str = None,
):
    """
    Train/fine-tune on a custom sequence.

    Args:
        config_file: Path to configuration file
        sequence: Sequence name
        gpu_id: GPU device ID
        online_depth: Online depth estimation method (None, 'DepthAnything', 'DepthAnythingV2-vitl')
        online_emb: Online embedding method (None, 'dinov2_vits14', 'dinov2_vits14_reg')
        checkpoint_path: Path to checkpoint file to resume from
    """
    # Load configuration
    seq_experiment = SourceFileLoader(
        os.path.basename(config_file), config_file
    ).load_module()

    config = seq_experiment.config

    # Update configuration with arguments
    config['data']['sequence'] = sequence
    if online_depth is not None:
        config['data']['online_depth'] = online_depth
    if online_emb is not None:
        config['data']['online_emb'] = online_emb

    # Create run name
    tracking_iters = config['tracking_obj']['num_iters']
    tracking_iters_init = config['tracking_obj']['num_iters_init']
    tracking_iters_cam = config['tracking_cam']['num_iters']
    online_depth_str = '' if config['data']['online_depth'] is None else '_' + config['data']['online_depth']
    online_emb_str = '' if config['data']['online_emb'] is None else '_' + config['data']['online_emb']
    run_name = f"{tracking_iters}_{tracking_iters_init}_{tracking_iters_cam}{online_depth_str}{online_emb_str}/{sequence}"

    config['run_name'] = run_name
    config['wandb']['name'] = run_name

    # Set GPU
    config['primary_device'] = f"cuda:{gpu_id}"

    # Create results directory
    results_dir = os.path.join(config["workdir"], run_name)
    os.makedirs(results_dir, exist_ok=True)

    # Save configuration
    config_save_path = os.path.join(results_dir, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_save_path}")

    # Set seed for reproducibility
    seed_everything(seed=config['seed'])

    # Check if already trained
    if os.path.isfile(os.path.join(results_dir, 'params.npz')):
        print(f"Experiment already completed: {run_name}")
        print(f"Results directory: {results_dir}")
        return results_dir

    # Load checkpoint if provided
    if checkpoint_path is not None:
        if os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            config['checkpoint'] = True
            config['checkpoint_path'] = checkpoint_path
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")

    # Initialize DynOMo model
    print(f"\n{'='*60}")
    print(f"Starting training for sequence: {sequence}")
    print(f"Results will be saved to: {results_dir}")
    print(f"{'='*60}\n")

    dynomo = DynOMo(config)

    # Run tracking/training
    dynomo.track()

    print(f"\n{'='*60}")
    print(f"Training completed for sequence: {sequence}")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}\n")

    return results_dir


def main():
    parser = argparse.ArgumentParser(
        description="Train/Fine-tune DynOMo on custom dataset"
    )

    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (e.g., configs/custom/dynomo_custom.py)"
    )
    parser.add_argument(
        "--sequence",
        type=str,
        required=True,
        help="Sequence name (folder name in basedir)"
    )

    # Optional arguments
    parser.add_argument(
        "--gpus",
        type=int,
        nargs='+',
        default=[0],
        help="GPU device IDs (default: [0])"
    )
    parser.add_argument(
        "--online_depth",
        type=str,
        default=None,
        choices=[None, 'DepthAnything', 'DepthAnythingV2-vitl'],
        help="Online depth estimation method (default: use precomputed depth)"
    )
    parser.add_argument(
        "--online_emb",
        type=str,
        default=None,
        choices=[None, 'dinov2_vits14', 'dinov2_vits14_reg'],
        help="Online embedding extraction method (default: use precomputed embeddings)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from"
    )

    args = parser.parse_args()

    # Get GPU ID
    gpu_id = args.gpus[0] if isinstance(args.gpus, list) else args.gpus

    # Train on sequence
    results_dir = train_custom_sequence(
        config_file=args.config,
        sequence=args.sequence,
        gpu_id=gpu_id,
        online_depth=args.online_depth,
        online_emb=args.online_emb,
        checkpoint_path=args.checkpoint,
    )

    print(f"\nTraining complete!")
    print(f"Results directory: {results_dir}")
    print(f"\nTo run inference, use:")
    print(f"python scripts/inference_custom.py --results_dir {results_dir}")


if __name__ == "__main__":
    main()
