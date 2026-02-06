"""
Custom Dataset for Online Tracking
Supports RGB, Depth, and SAM Segmentation data
"""

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from natsort import natsorted
import imageio
from sklearn.decomposition import PCA

from .basedataset import GradSLAMDataset


class CustomDataset(GradSLAMDataset):
    """
    Custom dataset loader for online tracking.

    Expected directory structure:
    basedir/
    ├── sequence_name/
    │   ├── rgb/          # RGB images
    │   ├── depth/        # Depth maps (optional if using online depth)
    │   └── sam/          # SAM segmentation masks

    Args:
        config_dict: Configuration dictionary
        basedir: Base directory containing the dataset
        sequence: Sequence name
        every_x_frame: Process every x frames
        start: Start frame index
        end: End frame index (-1 for all)
        desired_height: Target image height (as ratio or pixels)
        desired_width: Target image width (as ratio or pixels)
        load_embeddings: Whether to load embeddings
        embedding_dim: Embedding dimension
        online_depth: Online depth estimation method
        online_emb: Online embedding method
    """

    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        every_x_frame: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[float] = 1.0,
        desired_width: Optional[float] = 1.0,
        load_embeddings: Optional[bool] = False,
        embedding_dim: Optional[int] = 32,
        online_depth: Optional[str] = None,
        online_emb: Optional[str] = None,
        use_gt_poses: Optional[bool] = False,
        pose_file: Optional[str] = None,
        **kwargs,
    ):
        self.sequence = sequence
        self.input_folder = os.path.join(basedir, sequence)
        self.use_gt_poses = use_gt_poses
        self.pose_file = pose_file

        # Verify directories exist
        self.rgb_dir = os.path.join(self.input_folder, 'rgb')
        self.depth_dir = os.path.join(self.input_folder, 'depth')
        self.sam_dir = os.path.join(self.input_folder, 'sam')

        if not os.path.exists(self.rgb_dir):
            raise ValueError(f"RGB directory not found: {self.rgb_dir}")
        if not os.path.exists(self.sam_dir):
            raise ValueError(f"SAM directory not found: {self.sam_dir}")
        if online_depth is None and not os.path.exists(self.depth_dir):
            raise ValueError(f"Depth directory not found: {self.depth_dir}")

        # Initialize parent class
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

        print(f"Loaded custom dataset: {sequence}")
        print(f"  - RGB images: {len(self.color_paths)}")
        print(f"  - Depth maps: {len(self.depth_paths) if self.depth_paths else 'online'}")
        print(f"  - SAM masks: {len(self.instseg_paths)}")

    def read_embedding_from_file(self, embedding_path):
        """Read embedding from file."""
        embedding = np.load(embedding_path)
        if hasattr(self, 'embedding_downscale') and self.embedding_downscale is not None:
            shape = embedding.shape
            embedding = self.embedding_downscale.transform(embedding.reshape(-1, shape[2]))
            embedding = embedding.reshape((shape[0], shape[1], self.embedding_dim))
        return torch.from_numpy(embedding)

    def _load_bg(self, bg_path):
        """
        Load background mask from SAM segmentation.

        Args:
            bg_path: Path to SAM mask file

        Returns:
            Binary background mask (True for background, False for foreground)
        """
        # Load SAM mask - assuming it's a grayscale or RGB image
        # where 0 indicates background and non-zero indicates objects
        sam_mask = np.asarray(imageio.imread(bg_path), dtype=int)

        # If RGB, convert to grayscale by summing channels
        if len(sam_mask.shape) == 3:
            sam_mask = np.sum(sam_mask, axis=2)

        # Background is where mask is zero
        bg_mask = (sam_mask == 0)

        return bg_mask

    def _load_instseg(self, instseg_path):
        """
        Load instance segmentation from SAM.

        Args:
            instseg_path: Path to SAM segmentation file

        Returns:
            Instance segmentation map
        """
        instseg = np.asarray(imageio.imread(instseg_path), dtype=int)

        # If RGB, convert to single channel
        if len(instseg.shape) == 3:
            # Combine channels to create unique IDs for each segment
            # This assumes SAM outputs different colors for different instances
            instseg = instseg[:, :, 0] + instseg[:, :, 1] * 256 + instseg[:, :, 2] * 256 * 256

        return instseg

    def get_filepaths(self):
        """
        Get file paths for RGB, depth, embeddings, and segmentation masks.

        Returns:
            Tuple of (color_paths, depth_paths, embedding_paths, bg_paths, instseg_paths)
        """
        # Get RGB image paths
        rgb_extensions = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']
        color_paths = []
        for ext in rgb_extensions:
            color_paths.extend(glob.glob(f"{self.rgb_dir}/{ext}"))
        color_paths = natsorted(color_paths)

        if len(color_paths) == 0:
            raise ValueError(f"No RGB images found in {self.rgb_dir}")

        # Get depth paths
        if self.online_depth is None:
            depth_extensions = ['*.npy', '*.png', '*.exr']
            depth_paths = []
            for ext in depth_extensions:
                depth_paths.extend(glob.glob(f"{self.depth_dir}/{ext}"))
            depth_paths = natsorted(depth_paths)

            if len(depth_paths) != len(color_paths):
                raise ValueError(
                    f"Number of depth maps ({len(depth_paths)}) doesn't match "
                    f"number of RGB images ({len(color_paths)})"
                )
        else:
            depth_paths = None

        # Get SAM segmentation paths
        sam_extensions = ['*.png', '*.jpg', '*.npy']
        sam_paths = []
        for ext in sam_extensions:
            sam_paths.extend(glob.glob(f"{self.sam_dir}/{ext}"))
        sam_paths = natsorted(sam_paths)

        if len(sam_paths) != len(color_paths):
            raise ValueError(
                f"Number of SAM masks ({len(sam_paths)}) doesn't match "
                f"number of RGB images ({len(color_paths)})"
            )

        # Background paths are same as instance segmentation paths
        bg_paths = sam_paths
        instseg_paths = sam_paths

        # Get embedding paths if needed
        if self.load_embeddings and self.online_emb is None:
            embedding_dir = os.path.join(self.input_folder, 'embeddings')
            if os.path.exists(embedding_dir):
                embedding_paths = natsorted(glob.glob(f"{embedding_dir}/*.npy"))

                # Check if we need PCA for dimension reduction
                if len(embedding_paths) > 0:
                    features = np.load(embedding_paths[0])
                    if self.embedding_dim != features.shape[2]:
                        print(f"Applying PCA to reduce embeddings from {features.shape[2]} to {self.embedding_dim}")
                        pca = PCA(n_components=self.embedding_dim)
                        self.embedding_downscale = pca.fit(features.reshape(-1, features.shape[2]))
                    else:
                        self.embedding_downscale = None
                else:
                    embedding_paths = None
            else:
                print(f"Warning: Embeddings directory not found at {embedding_dir}")
                embedding_paths = None
        else:
            embedding_paths = None

        return color_paths, depth_paths, embedding_paths, bg_paths, instseg_paths

    def load_poses(self):
        """
        Load camera poses.

        If use_gt_poses is True and pose_file is provided, loads poses from file.
        Otherwise, returns identity poses.

        Returns:
            List of camera-to-world transformation matrices
        """
        poses = []

        if self.use_gt_poses and self.pose_file is not None:
            # Load poses from file
            pose_path = os.path.join(self.input_folder, self.pose_file)
            if os.path.exists(pose_path):
                if pose_path.endswith('.npy'):
                    # Load numpy array of poses
                    poses_array = np.load(pose_path)
                    for i in range(min(self.num_imgs, len(poses_array))):
                        c2w = torch.from_numpy(poses_array[i]).float()
                        poses.append(c2w)
                elif pose_path.endswith('.txt'):
                    # Load poses from text file (one 4x4 matrix per frame)
                    with open(pose_path, 'r') as f:
                        lines = f.readlines()

                    pose_data = []
                    for line in lines:
                        if line.strip():
                            pose_data.append([float(x) for x in line.strip().split()])

                    # Reshape into 4x4 matrices
                    num_poses = len(pose_data) // 4
                    for i in range(min(self.num_imgs, num_poses)):
                        pose_matrix = np.array(pose_data[i*4:(i+1)*4])
                        c2w = torch.from_numpy(pose_matrix).float()
                        poses.append(c2w)
                else:
                    print(f"Warning: Unknown pose file format: {pose_path}")
                    print("Using identity poses instead")

        # Fill remaining with identity poses
        while len(poses) < self.num_imgs:
            c2w = torch.eye(4).float()
            poses.append(c2w)

        # Store first pose world-to-camera for reference
        self.first_time_w2c = torch.linalg.inv(poses[0])

        return poses
