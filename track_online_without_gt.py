"""
Online Point Tracking with DynOMo (Without Ground Truth)

This script runs DynOMo online on custom datasets without ground truth:
1. Optimizes the scene frame-by-frame
2. Tracks user-specified query points through the sequence
3. Visualizes tracked trajectories

Usage:
    # Track with automatic query point selection (grid):
    python track_online_without_gt.py --config configs/custom/track_custom.py --sequence my_sequence

    # Track specific points:
    python track_online_without_gt.py --config configs/custom/track_custom.py --sequence my_sequence --query_points queries.json

Author: DynOMo Team
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path
import json
import cv2
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model.dynomo import DynOMo
from src.datasets.datasets.custom_test import CustomTestDataset
from src.utils.common_utils import seed_everything, save_params
from src.evaluate.trajectory_evaluator import TrajEvaluator, get_xy_grid
from src.utils.viz_utils import vis_trail, make_vid
from src.utils.gaussian_utils import unnormalize_points, normalize_points
from importlib.machinery import SourceFileLoader
import imageio


class OnlineTrackerWithoutGT:
    """
    Online point tracker for datasets without ground truth.

    Runs DynOMo optimization online and tracks query points through the sequence.
    """

    def __init__(self, config_path, sequence_name, output_dir, query_points_file=None):
        """
        Args:
            config_path: Path to configuration file
            sequence_name: Name of sequence to track
            output_dir: Directory to save results
            query_points_file: Optional JSON file with query points
        """
        self.sequence_name = sequence_name
        self.output_dir = output_dir
        self.query_points_file = query_points_file

        # Load configuration
        print(f"Loading config from {config_path}")
        config_module = SourceFileLoader(
            os.path.basename(config_path), config_path
        ).load_module()
        self.config = config_module.config

        # Update config with sequence name
        self.config['data']['sequence'] = sequence_name
        self.config['workdir'] = output_dir
        self.config['run_name'] = sequence_name

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to {output_dir}")

        # Set seed
        seed_everything(seed=self.config.get('seed', 0))

        # Save config
        config_save_path = os.path.join(output_dir, 'config.json')
        with open(config_save_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Config saved to {config_save_path}")

    def load_query_points(self, dataset):
        """
        Load or generate query points for tracking.

        Args:
            dataset: Dataset instance

        Returns:
            query_points: Tensor of shape (N, 2) with [x, y] coordinates
            query_info: Dict with metadata about queries
        """
        device = self.config['primary_device']

        if self.query_points_file and os.path.exists(self.query_points_file):
            # Load from file
            print(f"\nLoading query points from {self.query_points_file}")
            with open(self.query_points_file, 'r') as f:
                query_data = json.load(f)

            # Convert to tensor
            if 'points' in query_data:
                # Format: {"points": [[x1, y1], [x2, y2], ...]}
                query_points = torch.tensor(query_data['points'], dtype=torch.float32)
            elif 'normalized_points' in query_data:
                # Format: {"normalized_points": [[x1, y1], ...]}  # normalized [0, 1]
                normalized = torch.tensor(query_data['normalized_points'], dtype=torch.float32)
                query_points = unnormalize_points(
                    normalized,
                    dataset.desired_height,
                    dataset.desired_width,
                    do_round=False
                )
            else:
                raise ValueError("Query points file must contain 'points' or 'normalized_points'")

            query_info = {
                'source': 'file',
                'file': self.query_points_file,
                'num_points': len(query_points)
            }

        else:
            # Generate grid of query points
            print("\nGenerating grid of query points...")
            n_points = self.config.get('tracking', {}).get('n_query_points', 1024)

            # Use utility function to generate uniform grid
            query_points = get_xy_grid(
                H=dataset.desired_height,
                W=dataset.desired_width,
                N=n_points,
                B=1,
                device=device
            ).squeeze(0)  # Remove batch dimension

            query_info = {
                'source': 'grid',
                'num_points': len(query_points),
                'grid_size': int(np.sqrt(n_points))
            }

        query_points = query_points.to(device)
        print(f"Loaded {len(query_points)} query points")

        # Save query points for reference
        query_save_path = os.path.join(self.output_dir, 'query_points.json')
        query_normalized = normalize_points(
            query_points.cpu(),
            dataset.desired_height,
            dataset.desired_width
        )
        with open(query_save_path, 'w') as f:
            json.dump({
                'points': query_points.cpu().tolist(),
                'normalized_points': query_normalized.tolist(),
                'metadata': query_info
            }, f, indent=2)
        print(f"Query points saved to {query_save_path}")

        # Visualize query points on first frame
        self.visualize_query_points(dataset, query_points)

        return query_points, query_info

    def visualize_query_points(self, dataset, query_points):
        """Visualize query points on the first frame."""
        # Load first frame
        color, _, _, _, _, _, _ = dataset[0]

        # Convert to numpy
        img = color.cpu().permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)

        # Draw query points
        for point in query_points.cpu().numpy():
            x, y = int(point[0]), int(point[1])
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

        # Save
        query_vis_path = os.path.join(self.output_dir, 'query_points_frame0.png')
        cv2.imwrite(query_vis_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Query points visualization saved to {query_vis_path}")

    def run_online_tracking(self):
        """
        Run online DynOMo optimization and point tracking.
        """
        print("\n" + "=" * 60)
        print("Starting Online Tracking with DynOMo")
        print("=" * 60)

        # Initialize DynOMo
        print("\nInitializing DynOMo...")
        self.dynomo = DynOMo(self.config)

        # Run tracking (this runs online optimization)
        print("\nRunning online optimization...")
        print("This will optimize the scene frame-by-frame and may take a while.")
        self.dynomo.track()

        print("\n" + "=" * 60)
        print("Online optimization complete!")
        print("=" * 60)

    def extract_and_visualize_trajectories(self):
        """
        Extract trajectories for query points and visualize them.
        """
        print("\n" + "=" * 60)
        print("Extracting and Visualizing Trajectories")
        print("=" * 60)

        # Load optimized parameters
        params_path = os.path.join(self.output_dir, self.sequence_name, 'params.npz')
        if not os.path.exists(params_path):
            print(f"Warning: Optimized parameters not found at {params_path}")
            print("Make sure online tracking completed successfully.")
            return

        # Load query points
        dataset = self.dynomo.dataset
        query_points, query_info = self.load_query_points(dataset)

        # Initialize trajectory evaluator
        print("\nInitializing trajectory evaluator...")
        traj_evaluator = TrajEvaluator(
            config=self.config,
            params=None,  # Will be loaded from params_path
            results_dir=os.path.join(self.output_dir, self.sequence_name),
            vis_trajs=True,
            vis_thresh=self.config.get('tracking', {}).get('vis_thresh', 0.5),
            vis_thresh_start=self.config.get('tracking', {}).get('vis_thresh_start', 0.5),
            best_x=self.config.get('tracking', {}).get('best_x', 1),
            traj_len=self.config.get('tracking', {}).get('traj_len', 10),
            get_gauss_wise3D_track=True,
            queries_first_t=True,
            primary_device=self.config['primary_device']
        )

        # Extract trajectories
        print("\nTracking query points through sequence...")
        start_time = torch.zeros(len(query_points), dtype=torch.long).to(query_points.device)

        gs_traj_2D, gs_traj_3D, pred_visibility, gs_traj_2D_for_vis = traj_evaluator.get_gs_traj_pts(
            start_pixels=query_points,
            start_time=start_time,
            start_pixels_normalized=False,
            search_fg_only=False
        )

        print(f"Tracked trajectories shape: {gs_traj_2D.shape}")
        print(f"  - Number of points: {gs_traj_2D.shape[0]}")
        print(f"  - Number of frames: {gs_traj_2D.shape[1]}")

        # Save trajectories
        traj_save_path = os.path.join(self.output_dir, self.sequence_name, 'tracked_trajectories.npz')
        np.savez(
            traj_save_path,
            trajectories_2D=gs_traj_2D.cpu().numpy(),
            trajectories_3D=gs_traj_3D.cpu().numpy() if gs_traj_3D is not None else None,
            visibility=pred_visibility.cpu().numpy(),
            query_points=query_points.cpu().numpy(),
            query_info=query_info
        )
        print(f"Trajectories saved to {traj_save_path}")

        # Visualize trajectories
        self.visualize_trajectories(
            gs_traj_2D_for_vis if gs_traj_2D_for_vis is not None else gs_traj_2D,
            pred_visibility,
            traj_evaluator
        )

    def visualize_trajectories(self, trajectories_2D, visibility, traj_evaluator):
        """
        Create visualization videos of tracked points.

        Args:
            trajectories_2D: Tracked 2D trajectories
            visibility: Visibility predictions for each point
            traj_evaluator: TrajEvaluator instance for visualization
        """
        print("\nCreating trajectory visualizations...")

        # Prepare data in format expected by vis_trail
        # Load RGB frames for background
        dataset = self.dynomo.dataset
        num_frames = len(dataset)

        # Collect RGB frames
        rgb_frames = []
        for i in tqdm(range(num_frames), desc="Loading frames"):
            color, _, _, _, _, _, _ = dataset[i]
            rgb_np = color.cpu().permute(1, 2, 0).numpy()
            rgb_frames.append(rgb_np)
        rgb_frames = np.stack(rgb_frames)  # (T, H, W, 3)

        # Normalize trajectories for visualization
        h, w = dataset.desired_height, dataset.desired_width
        trajectories_normalized = normalize_points(
            trajectories_2D if len(trajectories_2D.shape) == 3 else trajectories_2D.squeeze(),
            h, w
        )

        # Create dummy occluded array (we don't have ground truth)
        occluded = torch.zeros(trajectories_normalized.shape[:-1], dtype=torch.bool)

        # Package data for visualization
        data = {
            'video': rgb_frames,
            'points': trajectories_normalized.cpu().numpy(),
            'occluded': occluded.cpu().numpy()
        }

        # Visualize with trails
        output_dir = os.path.join(self.output_dir, self.sequence_name, 'trajectory_visualization')
        os.makedirs(output_dir, exist_ok=True)

        vis_trail(
            output_dir,
            data,
            pred_visibility=(visibility > traj_evaluator.vis_thresh).cpu().numpy(),
            vis_traj=True,
            traj_len=self.config.get('tracking', {}).get('traj_len', 10),
            fps=self.config['data'].get('fps', 30)
        )

        print(f"Trajectory visualizations saved to {output_dir}")

    def visualize_grid_tracking(self):
        """
        Visualize dense grid tracking (optical flow-like visualization).
        """
        print("\n" + "=" * 60)
        print("Visualizing Dense Grid Tracking")
        print("=" * 60)

        # Load optimized parameters
        params_path = os.path.join(self.output_dir, self.sequence_name, 'params.npz')
        if not os.path.exists(params_path):
            print(f"Warning: Optimized parameters not found at {params_path}")
            return

        # Initialize trajectory evaluator with grid visualization
        traj_evaluator = TrajEvaluator(
            config=self.config,
            params=None,
            results_dir=os.path.join(self.output_dir, self.sequence_name),
            vis_trajs=True,
            vis_thresh=0.5,
            vis_thresh_start=0.5,
            best_x=1,
            traj_len=self.config.get('tracking', {}).get('traj_len', 10),
            get_gauss_wise3D_track=True,
            queries_first_t=True,
            primary_device=self.config['primary_device']
        )

        # Visualize grid
        print("Creating dense grid visualization...")
        traj_evaluator.vis_grid_trajs(mask=None, vis_vis_and_occ_same=False)

        print(f"Grid visualization saved to {os.path.join(self.output_dir, self.sequence_name, 'grid_points_vis')}")

    def visualize_optical_flow(self):
        """
        Visualize optical flow from tracked points.
        """
        print("\n" + "=" * 60)
        print("Visualizing Optical Flow")
        print("=" * 60)

        # Load optimized parameters
        params_path = os.path.join(self.output_dir, self.sequence_name, 'params.npz')
        if not os.path.exists(params_path):
            print(f"Warning: Optimized parameters not found at {params_path}")
            return

        # Initialize trajectory evaluator
        traj_evaluator = TrajEvaluator(
            config=self.config,
            params=None,
            results_dir=os.path.join(self.output_dir, self.sequence_name),
            primary_device=self.config['primary_device']
        )

        # Visualize flow
        print("Computing and visualizing optical flow...")
        traj_evaluator.vis_flow()

        print(f"Flow visualization saved to {os.path.join(self.output_dir, self.sequence_name, 'flow')}")

    def run(self, mode='full'):
        """
        Run the full tracking pipeline.

        Args:
            mode: One of:
                - 'full': Run optimization + tracking + visualization
                - 'optimize': Only run optimization
                - 'track': Only extract and visualize trajectories (assumes optimization done)
                - 'grid': Only visualize grid tracking
                - 'flow': Only visualize optical flow
        """
        if mode in ['full', 'optimize']:
            # Run online optimization
            self.run_online_tracking()

        if mode in ['full', 'track']:
            # Extract and visualize trajectories
            self.extract_and_visualize_trajectories()

        if mode == 'grid':
            # Visualize grid tracking
            self.visualize_grid_tracking()

        if mode == 'flow':
            # Visualize optical flow
            self.visualize_optical_flow()

        print("\n" + "=" * 60)
        print("Tracking Complete!")
        print("=" * 60)
        print(f"\nResults saved to: {self.output_dir}")
        print("\nOutput files:")
        print(f"  - Optimized scene: {self.output_dir}/{self.sequence_name}/params.npz")
        print(f"  - Query points: {self.output_dir}/query_points.json")
        print(f"  - Tracked trajectories: {self.output_dir}/{self.sequence_name}/tracked_trajectories.npz")
        print(f"  - Visualizations: {self.output_dir}/{self.sequence_name}/trajectory_visualization/")


def main():
    parser = argparse.ArgumentParser(
        description="Online point tracking with DynOMo (no ground truth required)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--sequence",
        type=str,
        required=True,
        help="Sequence name to track"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: results/tracking/<sequence>)"
    )
    parser.add_argument(
        "--query_points",
        type=str,
        default=None,
        help="JSON file with query points to track (optional, generates grid if not provided)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "optimize", "track", "grid", "flow"],
        help="Execution mode: full (optimize+track), optimize only, track only, grid vis, or flow vis"
    )

    args = parser.parse_args()

    # Set default output directory
    if args.output_dir is None:
        args.output_dir = os.path.join("results", "tracking", args.sequence)

    # Run tracker
    tracker = OnlineTrackerWithoutGT(
        config_path=args.config,
        sequence_name=args.sequence,
        output_dir=args.output_dir,
        query_points_file=args.query_points,
    )

    tracker.run(mode=args.mode)


if __name__ == "__main__":
    main()
