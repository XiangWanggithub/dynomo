"""
Custom Dataset for Testing without Ground Truth
This dataset supports RGB, depth (online or precomputed), and optional annotations.
For datasets without ground truth, use online depth prediction and dummy segmentation.
"""

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from natsort import natsorted
import imageio

from .basedataset import GradSLAMDataset


class CustomTestDataset(GradSLAMDataset):
    """
    Custom dataset for testing on new sequences without ground truth.

    Expected directory structure:
    basedir/sequence/
        rgb/            # RGB images (PNG or JPG)
            0000.png
            0001.png
            ...
        depth/          # [Optional] Precomputed depth maps (NPY format)
            0000.npy
            0001.npy
            ...
        poses.txt       # [Optional] Camera poses (4x4 matrices, one per line)
                        # If not provided, uses identity poses
        intrinsics.txt  # [Optional] Camera intrinsics (fx, fy, cx, cy)
                        # If not provided, uses config defaults

    Usage:
        # With online depth prediction (no depth folder needed):
        dataset = CustomTestDataset(
            config_dict=config,
            basedir="/path/to/data",
            sequence="my_sequence",
            online_depth="DepthAnythingV2",  # or "DepthAnything"
            load_embeddings=False,
        )

        # With precomputed depth:
        dataset = CustomTestDataset(
            config_dict=config,
            basedir="/path/to/data",
            sequence="my_sequence",
            online_depth=None,  # Use precomputed depth
            load_embeddings=False,
        )
    """

    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        every_x_frame: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dim: Optional[int] = 512,
        online_depth: Optional[str] = None,  # "DepthAnything", "DepthAnythingV2", or None
        online_emb: Optional[str] = None,
        use_dummy_segmentation: Optional[bool] = True,  # Use dummy masks when no annotation
        **kwargs,
    ):
        """
        Args:
            config_dict: Configuration dictionary with camera parameters
            basedir: Base directory containing sequences
            sequence: Sequence name
            every_x_frame: Sample every x frames
            start: Start frame index
            end: End frame index (-1 for all frames)
            desired_height: Target height for resizing
            desired_width: Target width for resizing
            load_embeddings: Whether to load/compute feature embeddings
            embedding_dim: Dimension of feature embeddings
            online_depth: Online depth prediction method ("DepthAnything", "DepthAnythingV2", or None)
            online_emb: Online embedding computation method
            use_dummy_segmentation: If True, creates dummy segmentation masks
        """
        self.input_folder = os.path.join(basedir, sequence)
        self.use_dummy_segmentation = use_dummy_segmentation

        # Check if RGB folder exists
        if not os.path.exists(os.path.join(self.input_folder, "rgb")):
            raise ValueError(f"RGB folder not found at {os.path.join(self.input_folder, 'rgb')}")

        # Initialize base class
        super().__init__(
            config_dict,
            every_x_frame=every_x_frame,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dim=embedding_dim,
            online_depth=online_depth,
            online_emb=online_emb,
            **kwargs,
        )

        print(f"Loaded CustomTestDataset with {len(self.color_paths)} frames")
        print(f"  - RGB source: {self.input_folder}/rgb/")
        print(f"  - Depth mode: {'Online (' + str(online_depth) + ')' if online_depth else 'Precomputed'}")
        print(f"  - Segmentation: {'Dummy masks' if use_dummy_segmentation else 'From files'}")
        print(f"  - Embeddings: {'Online (' + str(online_emb) + ')' if online_emb else ('Precomputed' if load_embeddings else 'Disabled')}")

    def get_filepaths(self):
        """
        Collect file paths for RGB, depth, segmentation, and embeddings.
        Returns paths compatible with base class __getitem__ method.
        """
        # RGB images (required)
        rgb_folder = os.path.join(self.input_folder, "rgb")
        color_paths = natsorted(glob.glob(f"{rgb_folder}/*.png") + glob.glob(f"{rgb_folder}/*.jpg"))

        if len(color_paths) == 0:
            raise ValueError(f"No images found in {rgb_folder}")

        # Apply start/end slicing
        if self.end == -1:
            color_paths = color_paths[self.start:]
        else:
            color_paths = color_paths[self.start:self.end]

        num_imgs = len(color_paths)

        # Depth paths (optional if using online depth)
        if self.online_depth is None:
            depth_folder = os.path.join(self.input_folder, "depth")
            if os.path.exists(depth_folder):
                depth_paths = natsorted(glob.glob(f"{depth_folder}/*.npy"))
                if self.end == -1:
                    depth_paths = depth_paths[self.start:]
                else:
                    depth_paths = depth_paths[self.start:self.end]

                if len(depth_paths) != num_imgs:
                    print(f"Warning: Found {len(depth_paths)} depth files but {num_imgs} RGB images")
                    print("         Consider using online_depth='DepthAnythingV2' instead")
            else:
                print(f"Warning: No depth folder found at {depth_folder}")
                print("         Consider using online_depth='DepthAnythingV2'")
                depth_paths = None
        else:
            depth_paths = None

        # Background and instance segmentation paths
        if not self.use_dummy_segmentation:
            seg_folder = os.path.join(self.input_folder, "segmentation")
            if os.path.exists(seg_folder):
                bg_paths = natsorted(glob.glob(f"{seg_folder}/*.png"))
                instseg_paths = natsorted(glob.glob(f"{seg_folder}/*.png"))

                if self.end == -1:
                    bg_paths = bg_paths[self.start:]
                    instseg_paths = instseg_paths[self.start:]
                else:
                    bg_paths = bg_paths[self.start:self.end]
                    instseg_paths = instseg_paths[self.start:self.end]
            else:
                print(f"Warning: Segmentation folder not found, using dummy masks")
                bg_paths = [None] * num_imgs
                instseg_paths = [None] * num_imgs
        else:
            # Use dummy paths (will be handled in _load_bg and _load_instseg)
            bg_paths = [None] * num_imgs
            instseg_paths = [None] * num_imgs

        # Embedding paths (optional if using online embeddings)
        if self.load_embeddings and self.online_emb is None:
            emb_folder = os.path.join(self.input_folder, "embeddings")
            if os.path.exists(emb_folder):
                embedding_paths = natsorted(glob.glob(f"{emb_folder}/*.npy"))
                if self.end == -1:
                    embedding_paths = embedding_paths[self.start:]
                else:
                    embedding_paths = embedding_paths[self.start:self.end]
            else:
                print(f"Warning: Embedding folder not found, consider using online_emb")
                embedding_paths = None
        else:
            embedding_paths = None

        return color_paths, depth_paths, embedding_paths, bg_paths, instseg_paths

    def _load_bg(self, bg_path):
        """
        Load background mask. Returns binary mask (True = background, False = foreground).
        If bg_path is None, returns a dummy mask (all background).
        """
        if bg_path is None or not os.path.exists(bg_path):
            # Dummy background mask - assume everything is background
            # This will be resized by the base class preprocessing
            return np.ones((self.orig_height, self.orig_width), dtype=bool)

        # Load from file
        bg = np.asarray(imageio.imread(bg_path), dtype=int)
        if len(bg.shape) == 3:
            bg = bg[:, :, 0]  # Take first channel if RGB

        # Convert to binary mask
        bg = bg == 0  # Assumes 0 = background, non-zero = foreground
        return bg

    def _load_instseg(self, instseg_path):
        """
        Load instance segmentation. Returns integer label map.
        If instseg_path is None, returns a dummy segmentation (all same instance).
        """
        if instseg_path is None or not os.path.exists(instseg_path):
            # Dummy instance segmentation - all pixels belong to instance 0
            return np.zeros((self.orig_height, self.orig_width), dtype=int)

        # Load from file
        instseg = np.asarray(imageio.imread(instseg_path), dtype=int)
        if len(instseg.shape) == 3:
            instseg = instseg[:, :, 0]  # Take first channel if RGB

        return instseg

    def read_embedding_from_file(self, embedding_path):
        """Load embedding from NPY file."""
        embedding = np.load(embedding_path)
        return torch.from_numpy(embedding).float()

    def load_poses(self):
        """
        Load camera poses from poses.txt file.
        Format: Each line is 16 values (4x4 matrix in row-major order)

        If file doesn't exist, returns identity poses for all frames.
        """
        poses_file = os.path.join(self.input_folder, "poses.txt")

        if os.path.exists(poses_file):
            try:
                poses = []
                with open(poses_file, 'r') as f:
                    for line in f:
                        values = [float(x) for x in line.strip().split()]
                        if len(values) == 16:
                            pose = np.array(values).reshape(4, 4)
                            poses.append(pose)

                if len(poses) > 0:
                    print(f"Loaded {len(poses)} camera poses from {poses_file}")
                    return np.stack(poses)
            except Exception as e:
                print(f"Warning: Failed to load poses from {poses_file}: {e}")

        # Return identity poses
        print("Using identity poses (static camera assumption)")
        num_frames = len(self.color_paths)
        identity = np.eye(4, dtype=np.float32)
        return np.stack([identity] * num_frames)

    def load_intrinsics(self):
        """
        Load camera intrinsics from intrinsics.txt file.
        Format: fx fy cx cy

        If file doesn't exist, uses values from config_dict.
        """
        intrinsics_file = os.path.join(self.input_folder, "intrinsics.txt")

        if os.path.exists(intrinsics_file):
            try:
                with open(intrinsics_file, 'r') as f:
                    line = f.readline().strip()
                    fx, fy, cx, cy = [float(x) for x in line.split()]
                    print(f"Loaded intrinsics from {intrinsics_file}: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
                    return fx, fy, cx, cy
            except Exception as e:
                print(f"Warning: Failed to load intrinsics from {intrinsics_file}: {e}")

        # Use config defaults
        print("Using intrinsics from config")
        return None
