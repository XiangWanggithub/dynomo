"""
Test/Inference Script for Datasets Without Ground Truth

This script demonstrates how to:
1. Load a custom dataset without ground truth
2. Run inference using a trained DynOMo model
3. Visualize results (rendered RGB, depth, etc.) without computing metrics

Usage:
    python test_without_gt.py --config configs/custom/test_config.py --sequence my_sequence

Author: DynOMo Team
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path
import imageio
import cv2
from tqdm import tqdm
import copy

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.datasets.datasets.custom_test import CustomTestDataset
from src.model.dynomo import DynOMo
from src.utils.camera_helpers import setup_camera
from src.utils.viz_utils import make_vid
from importlib.machinery import SourceFileLoader


class TestWithoutGroundTruth:
    """
    Test runner for datasets without ground truth.
    Focuses on visualization and qualitative assessment.
    """

    def __init__(self, config_path, sequence_name, output_dir, checkpoint_path=None):
        """
        Args:
            config_path: Path to configuration file
            sequence_name: Name of sequence to test
            output_dir: Directory to save results
            checkpoint_path: Path to trained model checkpoint (optional)
        """
        self.sequence_name = sequence_name
        self.output_dir = output_dir
        self.checkpoint_path = checkpoint_path

        # Load configuration
        print(f"Loading config from {config_path}")
        self.config = SourceFileLoader(
            os.path.basename(config_path), config_path
        ).load_module().config

        # Update config with sequence name
        self.config['data']['sequence'] = sequence_name

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to {output_dir}")

    def load_dataset(self):
        """Load the custom test dataset."""
        print("\n" + "=" * 50)
        print("Loading Dataset")
        print("=" * 50)

        # Extract dataset configuration
        data_config = self.config['data']

        # Create dataset
        self.dataset = CustomTestDataset(
            config_dict=data_config,
            basedir=data_config['basedir'],
            sequence=self.sequence_name,
            every_x_frame=data_config.get('every_x_frame', 1),
            start=data_config.get('start', 0),
            end=data_config.get('end', -1),
            desired_height=data_config['desired_height'],
            desired_width=data_config['desired_width'],
            load_embeddings=data_config.get('load_embeddings', False),
            embedding_dim=data_config.get('embedding_dim', 512),
            online_depth=data_config.get('online_depth', 'DepthAnythingV2'),
            online_emb=data_config.get('online_emb', None),
            use_dummy_segmentation=True,  # Use dummy masks for datasets without annotations
        )

        print(f"\nDataset loaded successfully: {len(self.dataset)} frames")
        return self.dataset

    def load_model(self):
        """Load the trained DynOMo model."""
        if self.checkpoint_path is None:
            print("\nNo checkpoint provided - skipping model loading")
            print("(This script will only visualize dataset loading)")
            return None

        print("\n" + "=" * 50)
        print("Loading Model")
        print("=" * 50)

        if not os.path.exists(self.checkpoint_path):
            raise ValueError(f"Checkpoint not found: {self.checkpoint_path}")

        print(f"Loading checkpoint from {self.checkpoint_path}")

        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DynOMo(self.config, device)

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=device)

        # Load model parameters
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'params' in checkpoint:
            # Some checkpoints store params directly
            self.model.params = checkpoint['params']
        else:
            print("Warning: Checkpoint format not recognized, attempting direct load")
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print("Model loaded successfully")

        return self.model

    def visualize_dataset(self):
        """
        Visualize the loaded dataset without running inference.
        Useful for verifying data loading is correct.
        """
        print("\n" + "=" * 50)
        print("Visualizing Dataset (No Inference)")
        print("=" * 50)

        # Create output directories
        rgb_dir = os.path.join(self.output_dir, "input_rgb")
        depth_dir = os.path.join(self.output_dir, "input_depth")
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

        num_frames = len(self.dataset)
        print(f"Processing {num_frames} frames...")

        for idx in tqdm(range(num_frames), desc="Saving frames"):
            # Load data from dataset
            color, depth, intrinsics, pose, embeddings, bg, instseg = self.dataset[idx]

            # Save RGB
            rgb_np = color.cpu().permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
            rgb_np = (rgb_np * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(rgb_dir, f"{idx:04d}.png"),
                cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            )

            # Save depth (colorized)
            depth_np = depth[0].cpu().numpy()  # (1, H, W) -> (H, W)
            depth_valid = depth_np[depth_np > 0]

            if len(depth_valid) > 0:
                vmin, vmax = np.percentile(depth_valid, [2, 98])
            else:
                vmin, vmax = 0, 1

            depth_normalized = np.clip((depth_np - vmin) / (vmax - vmin + 1e-10), 0, 1)
            depth_colored = cv2.applyColorMap(
                (depth_normalized * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            cv2.imwrite(os.path.join(depth_dir, f"{idx:04d}.png"), depth_colored)

        # Create videos
        print("\nCreating videos...")
        fps = self.config['data'].get('fps', 30)
        make_vid(rgb_dir, fps)
        make_vid(depth_dir, fps)

        print(f"\nVisualization complete!")
        print(f"  - RGB frames: {rgb_dir}")
        print(f"  - Depth frames: {depth_dir}")
        print(f"  - Videos: {rgb_dir}/video.mp4, {depth_dir}/video.mp4")

    def run_inference(self):
        """
        Run inference with trained model (if loaded).
        Generates rendered RGB, depth, and other outputs.
        """
        if self.model is None:
            print("\nNo model loaded - skipping inference")
            return

        print("\n" + "=" * 50)
        print("Running Inference")
        print("=" * 50)

        # Create output directories
        render_rgb_dir = os.path.join(self.output_dir, "rendered_rgb")
        render_depth_dir = os.path.join(self.output_dir, "rendered_depth")
        os.makedirs(render_rgb_dir, exist_ok=True)
        os.makedirs(render_depth_dir, exist_ok=True)

        num_frames = len(self.dataset)
        device = next(self.model.parameters()).device

        print(f"Processing {num_frames} frames...")

        with torch.no_grad():
            for idx in tqdm(range(num_frames), desc="Rendering"):
                # Load data
                color, depth, intrinsics, pose, embeddings, bg, instseg = self.dataset[idx]

                # Setup camera
                intrinsics_3x3 = intrinsics[:3, :3]
                w2c = torch.linalg.inv(pose)

                cam = setup_camera(
                    color.shape[2],  # width
                    color.shape[1],  # height
                    intrinsics_3x3.cpu().numpy(),
                    w2c.cpu().numpy(),
                    device=device
                )

                # Prepare current frame data
                curr_data = {
                    'cam': cam,
                    'im': color.to(device),
                    'depth': depth.to(device),
                    'id': idx,
                    'intrinsics': intrinsics.to(device),
                    'w2c': w2c.to(device),
                }

                # Render (this is a placeholder - actual rendering depends on your model)
                # You'll need to adapt this based on your DynOMo model's interface
                try:
                    rendered_rgb, rendered_depth = self.model.render(curr_data)

                    # Save rendered RGB
                    rgb_np = rendered_rgb.cpu().permute(1, 2, 0).numpy()
                    rgb_np = np.clip(rgb_np * 255, 0, 255).astype(np.uint8)
                    cv2.imwrite(
                        os.path.join(render_rgb_dir, f"{idx:04d}.png"),
                        cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
                    )

                    # Save rendered depth
                    depth_np = rendered_depth[0].cpu().numpy()
                    depth_valid = depth_np[depth_np > 0]

                    if len(depth_valid) > 0:
                        vmin, vmax = np.percentile(depth_valid, [2, 98])
                    else:
                        vmin, vmax = 0, 1

                    depth_normalized = np.clip((depth_np - vmin) / (vmax - vmin + 1e-10), 0, 1)
                    depth_colored = cv2.applyColorMap(
                        (depth_normalized * 255).astype(np.uint8),
                        cv2.COLORMAP_JET
                    )
                    cv2.imwrite(os.path.join(render_depth_dir, f"{idx:04d}.png"), depth_colored)

                except Exception as e:
                    print(f"\nWarning: Rendering failed for frame {idx}: {e}")
                    print("This is expected if the model interface differs from assumptions")
                    print("Please adapt the rendering code to match your model's API")
                    break

        # Create videos
        print("\nCreating videos...")
        fps = self.config['data'].get('fps', 30)
        make_vid(render_rgb_dir, fps)
        make_vid(render_depth_dir, fps)

        print(f"\nInference complete!")
        print(f"  - Rendered RGB: {render_rgb_dir}")
        print(f"  - Rendered depth: {render_depth_dir}")

    def run(self, visualize_only=False):
        """
        Run the full test pipeline.

        Args:
            visualize_only: If True, only visualize dataset without inference
        """
        # Load dataset
        self.load_dataset()

        # Visualize dataset
        self.visualize_dataset()

        # Run inference if model provided and not visualize_only
        if not visualize_only:
            self.load_model()
            self.run_inference()

        print("\n" + "=" * 50)
        print("Test Complete!")
        print("=" * 50)
        print(f"All results saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Test DynOMo on datasets without ground truth"
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
        help="Sequence name to test"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: results/<sequence>)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (if not provided, only visualizes dataset)"
    )
    parser.add_argument(
        "--visualize_only",
        action="store_true",
        help="Only visualize dataset without running inference"
    )

    args = parser.parse_args()

    # Set default output directory
    if args.output_dir is None:
        args.output_dir = os.path.join("results", "test_without_gt", args.sequence)

    # Run test
    tester = TestWithoutGroundTruth(
        config_path=args.config,
        sequence_name=args.sequence,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
    )

    tester.run(visualize_only=args.visualize_only)


if __name__ == "__main__":
    main()
