from src.datasets.datasets import (
    load_dataset_config,
    CustomDataset,
    datautils
)
import numpy as np
import os
import torch
from src.utils.camera_helpers import as_intrinsics_matrix


def get_dataset(
    config_dict,
    basedir,
    sequence,
    **kwargs):
    """
    Get dataset instance based on configuration.

    Args:
        config_dict: Dataset configuration dictionary
        basedir: Base directory for data
        sequence: Sequence name
        **kwargs: Additional arguments

    Returns:
        CustomDataset instance
    """
    if config_dict["dataset_name"].lower() in ["custom"]:
        return CustomDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def get_data(config):
    """
    Load dataset based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Dataset instance
    """
    dataset_config = config["data"]
    gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])

    # Poses are relative to the first frame
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        relative_pose=True,
        device=config["primary_device"],
        basedir=dataset_config["basedir"],
        sequence=dataset_config["sequence"],
        every_x_frame=dataset_config["every_x_frame"],
        start=dataset_config["start"],
        end=dataset_config["end"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        load_embeddings=dataset_config["load_embeddings"],
        depth_type=dataset_config.get('depth_type', None),
        cam_type=dataset_config.get('cam_type', None),
        embedding_dim=dataset_config["embedding_dim"],
        start_from_complete_pc=dataset_config["start_from_complete_pc"],
        novel_view_mode=dataset_config['novel_view_mode'],
        factor=dataset_config.get('factor', 2),
        do_scale=dataset_config.get('do_scale', False),
        online_emb=dataset_config['online_emb'],
        online_depth=dataset_config['online_depth'],
        use_gt_poses=dataset_config.get('use_gt_poses', False),
        pose_file=dataset_config.get('pose_file', None)
    )

    return dataset


def get_cam_data(config, orig_image_size=False):
    """
    Get camera intrinsics and pose data.

    Args:
        config: Configuration dictionary
        orig_image_size: Whether to use original image size

    Returns:
        Tuple of (intrinsics, pose, height, width)
    """
    config_dict = load_dataset_config(config['data']["gradslam_data_cfg"])

    if orig_image_size:
        desired_image_height = config_dict["camera_params"]["image_height"]
        desired_image_width = config_dict["camera_params"]["image_width"]
    else:
        desired_image_height = config['data']['desired_image_height']
        desired_image_width = config['data']['desired_image_width']

    orig_height = config_dict["camera_params"]["image_height"]
    orig_width = config_dict["camera_params"]["image_width"]
    fx = config_dict["camera_params"]["fx"]
    fy = config_dict["camera_params"]["fy"]
    cx = config_dict["camera_params"]["cx"]
    cy = config_dict["camera_params"]["cy"]

    height_downsample_ratio = float(desired_image_height) / orig_height
    width_downsample_ratio = float(desired_image_width) / orig_width

    K = as_intrinsics_matrix([fx, fy, cx, cy])
    K = torch.from_numpy(K)
    K = datautils.scale_intrinsics(K, height_downsample_ratio, width_downsample_ratio)
    intrinsics = torch.eye(4).to(K)
    intrinsics[:3, :3] = K

    pose = torch.eye(4).float()

    return intrinsics, pose, desired_image_height, desired_image_width


def load_scene_data(config=None, results_dir=None, device="cuda:0", file=None):
    """
    Load saved scene parameters.

    Args:
        config: Configuration dictionary
        results_dir: Results directory path
        device: Device to load parameters to
        file: Optional specific file path

    Returns:
        Tuple of (params, timestep, intrinsics, w2c)
    """
    if file is None:
        params = dict(np.load(f"{results_dir}/params.npz", allow_pickle=True))
    else:
        params = dict(np.load(file, allow_pickle=True))

    _params = dict()
    for k, v in params.items():
        if (v != np.array(None)).all():
            _params[k] = torch.tensor(v).to(device).float()
        else:
            _params[k] = None
    params = _params

    if "timestep" in params.keys():
        return params, params['timestep'], params['intrinsics'], params['w2c']
    else:
        params['means3D'] = params['means3D'].permute(1, 2, 0)[:, :, 2:]
        params['unnorm_rotations'] = params['unnorm_rotations'].permute(1, 2, 0)[:, :, 2:]
        params['bg'] = params['seg_colors'][:, 2]
        params['rgb_colors'] = params['rgb_colors'].permute(1, 2, 0)[:, :, 0]
        params['timestep'] = torch.zeros(params['means3D'].shape[0])
        cam_rots = np.tile([1, 0, 0, 0], (1, 1))
        cam_rots = torch.from_numpy(np.tile(cam_rots[:, :, None], (1, 1, params['means3D'].shape[2])))
        params['cam_unnorm_rots'] = cam_rots.to(params['means3D'].device).float()
        params['cam_trans'] = torch.from_numpy(np.zeros((1, 3, params['means3D'].shape[2]))).to(params['means3D'].device).float()

        return params, None, None, None
