#!/usr/bin/env python3
"""
Inference Script for Custom Dataset Online Tracking

This script performs inference on trained models for custom datasets.

Usage:
    # Using results directory
    python scripts/inference_custom.py \
        --results_dir experiments/custom/200_200_200/your_sequence

    # Using checkpoint file
    python scripts/inference_custom.py \
        --checkpoint experiments/custom/200_200_200/your_sequence/params.npz \
        --config configs/custom/dynomo_custom.py \
        --sequence your_sequence

Optional arguments:
    --novel_view_mode circle      # Novel view synthesis mode ('circle', 'zoom_out')
    --vis_trajs                   # Visualize trajectories
    --vis_grid                    # Visualize evaluation grids
    --eval_renderings             # Evaluate rendering quality
    --eval_trajs                  # Evaluate trajectory quality
    --gpu 0                       # GPU device ID
"""

import os
import sys
import argparse
import json
from importlib.machinery import SourceFileLoader

# Add parent directory to path
sys.path.append(os.getcwd())

from src.utils.common_utils import seed_everything
from src.model.dynomo import DynOMo


def inference_custom_sequence(
    results_dir: str = None,
    checkpoint_file: str = None,
    config_file: str = None,
    sequence: str = None,
    gpu_id: int = 0,
    novel_view_mode: str = None,
    eval_renderings: bool = True,
    eval_trajs: bool = True,
    vis_trajs: bool = True,
    vis_grid: bool = True,
    vis_fg_only: bool = True,
    vis_gt: bool = False,
    vis_all: bool = False,
    best_x: int = 1,
    alpha_traj: bool = False,
    traj_len: int = 20,
):
    """
    Run inference on a trained model.

    Args:
        results_dir: Path to results directory (contains config.json and params.npz)
        checkpoint_file: Path to checkpoint file (params.npz)
        config_file: Path to configuration file (if using checkpoint_file)
        sequence: Sequence name (if using checkpoint_file)
        gpu_id: GPU device ID
        novel_view_mode: Novel view synthesis mode ('circle', 'zoom_out', None)
        eval_renderings: Whether to evaluate rendering quality
        eval_trajs: Whether to evaluate trajectory quality
        vis_trajs: Whether to visualize trajectories
        vis_grid: Whether to visualize evaluation grids
        vis_fg_only: Whether to visualize only foreground
        vis_gt: Whether to visualize ground truth
        vis_all: Whether to visualize all renderings
        best_x: Oracle result, get best Gaussian out of x
        alpha_traj: Whether to use alpha blending for trajectory
        traj_len: Trajectory length for visualization
    """

    # Load configuration
    if results_dir is not None:
        # Load from results directory
        if not os.path.exists(results_dir):
            raise ValueError(f"Results directory not found: {results_dir}")

        config_path = os.path.join(results_dir, 'config.json')
        if not os.path.exists(config_path):
            raise ValueError(f"Configuration file not found: {config_path}")

        checkpoint_path = os.path.join(results_dir, 'params.npz')
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint file not found: {checkpoint_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        print(f"Loaded configuration from: {config_path}")
        print(f"Loaded checkpoint from: {checkpoint_path}")

    elif checkpoint_file is not None and config_file is not None:
        # Load from checkpoint and config files
        if not os.path.exists(checkpoint_file):
            raise ValueError(f"Checkpoint file not found: {checkpoint_file}")
        if not os.path.exists(config_file):
            raise ValueError(f"Configuration file not found: {config_file}")

        seq_experiment = SourceFileLoader(
            os.path.basename(config_file), config_file
        ).load_module()

        config = seq_experiment.config

        if sequence is not None:
            config['data']['sequence'] = sequence

        checkpoint_path = checkpoint_file

        print(f"Loaded configuration from: {config_file}")
        print(f"Loaded checkpoint from: {checkpoint_file}")

    else:
        raise ValueError(
            "Must provide either --results_dir or both --checkpoint and --config"
        )

    # Update configuration for inference
    config['primary_device'] = f"cuda:{gpu_id}"
    config['just_eval'] = True
    config['checkpoint'] = True

    # Update visualization settings
    config['viz']['vis_trajs'] = vis_trajs
    config['viz']['vis_grid'] = vis_grid
    config['viz']['vis_all'] = vis_all
    config['viz']['vis_gt'] = vis_gt
    config['viz']['vis_fg_only'] = vis_fg_only

    # Set seed for reproducibility
    seed_everything(seed=config['seed'])

    # Initialize DynOMo model
    print(f"\n{'='*60}")
    print(f"Starting inference for sequence: {config['data']['sequence']}")
    print(f"{'='*60}\n")

    dynomo = DynOMo(config)

    # Run inference
    dynomo.eval(
        novel_view_mode=novel_view_mode,
        eval_renderings=eval_renderings,
        eval_trajs=eval_trajs,
        vis_trajs=vis_trajs,
        vis_grid=vis_grid,
        vis_fg_only=vis_fg_only,
        best_x=best_x,
        alpha_traj=alpha_traj,
        traj_len=traj_len,
    )

    print(f"\n{'='*60}")
    print(f"Inference completed!")
    print(f"Results saved to: {dynomo.eval_dir}")
    print(f"{'='*60}\n")

    return dynomo.eval_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on trained DynOMo model"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--results_dir",
        type=str,
        help="Path to results directory (contains config.json and params.npz)"
    )
    input_group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint file (params.npz)"
    )

    # Additional arguments when using --checkpoint
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (required if using --checkpoint)"
    )
    parser.add_argument(
        "--sequence",
        type=str,
        help="Sequence name (optional, overrides config)"
    )

    # GPU settings
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID (default: 0)"
    )

    # Evaluation settings
    parser.add_argument(
        "--novel_view_mode",
        type=str,
        default=None,
        choices=[None, 'circle', 'zoom_out'],
        help="Novel view synthesis mode"
    )
    parser.add_argument(
        "--no_eval_renderings",
        action="store_false",
        dest="eval_renderings",
        help="Disable rendering evaluation"
    )
    parser.add_argument(
        "--no_eval_trajs",
        action="store_false",
        dest="eval_trajs",
        help="Disable trajectory evaluation"
    )

    # Visualization settings
    parser.add_argument(
        "--no_vis_trajs",
        action="store_false",
        dest="vis_trajs",
        help="Disable trajectory visualization"
    )
    parser.add_argument(
        "--no_vis_grid",
        action="store_false",
        dest="vis_grid",
        help="Disable grid visualization"
    )
    parser.add_argument(
        "--vis_bg_and_fg",
        action="store_false",
        dest="vis_fg_only",
        help="Visualize both background and foreground"
    )
    parser.add_argument(
        "--vis_gt",
        action="store_true",
        help="Visualize ground truth"
    )
    parser.add_argument(
        "--vis_all",
        action="store_true",
        help="Visualize all renderings"
    )

    # Advanced settings
    parser.add_argument(
        "--best_x",
        type=int,
        default=1,
        help="Oracle result: get best Gaussian out of x (default: 1)"
    )
    parser.add_argument(
        "--alpha_traj",
        action="store_true",
        help="Use alpha blending for trajectory visualization"
    )
    parser.add_argument(
        "--traj_len",
        type=int,
        default=20,
        help="Trajectory length for visualization (default: 20)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.checkpoint is not None and args.config is None:
        parser.error("--config is required when using --checkpoint")

    # Run inference
    eval_dir = inference_custom_sequence(
        results_dir=args.results_dir,
        checkpoint_file=args.checkpoint,
        config_file=args.config,
        sequence=args.sequence,
        gpu_id=args.gpu,
        novel_view_mode=args.novel_view_mode,
        eval_renderings=args.eval_renderings,
        eval_trajs=args.eval_trajs,
        vis_trajs=args.vis_trajs,
        vis_grid=args.vis_grid,
        vis_fg_only=args.vis_fg_only,
        vis_gt=args.vis_gt,
        vis_all=args.vis_all,
        best_x=args.best_x,
        alpha_traj=args.alpha_traj,
        traj_len=args.traj_len,
    )

    print(f"\nInference complete!")
    print(f"Evaluation results saved to: {eval_dir}")


if __name__ == "__main__":
    main()
