"""
Configuration for Online Tracking on Custom Datasets (Without Ground Truth)

This config is used for running DynOMo online optimization and point tracking
on datasets without ground truth depth or annotations.

Usage:
    python track_online_without_gt.py --config configs/custom/track_custom.py --sequence my_sequence
"""

config = {
    # Random seed for reproducibility
    'seed': 0,

    # Primary device
    'primary_device': 'cuda:0',  # or 'cuda:1', 'cpu', etc.

    # Working directory (will be set automatically)
    'workdir': 'results/tracking',
    'run_name': 'custom_sequence',

    # Checkpointing
    'checkpoint': False,  # Set to True to resume from checkpoint
    'save_checkpoints': True,
    'checkpoint_interval': 50,  # Save checkpoint every N frames

    # Evaluation flags
    'just_eval': False,  # Set to True to only evaluate existing results
    'eval_during': False,  # Whether to evaluate during training

    # Weights & Biases logging
    'use_wandb': False,  # Set to True to enable WandB logging
    'wandb': {
        'project': 'dynomo_custom',
        'group': 'tracking',
        'name': 'custom_sequence'
    },

    # ===========================
    # Dataset Configuration
    # ===========================
    'data': {
        'dataset_name': 'custom_test',
        'gradslam_data_cfg': 'custom_test',  # Used for trajectory evaluator

        # Data paths
        'basedir': '/path/to/your/data',  # UPDATE THIS
        'sequence': 'test_sequence',       # Will be overridden by command line

        # Frame sampling
        'every_x_frame': 1,  # Process every x frames
        'start': 0,
        'end': -1,  # -1 for all frames

        # Image dimensions
        'desired_height': 360,  # Reduce if OOM
        'desired_width': 640,
        'camera_params': {
            'image_height': 360,
            'image_width': 640,
            'fx': 525.0,
            'fy': 525.0,
            'cx': 320.0,
            'cy': 240.0,
            'png_depth_scale': 1.0,
        },

        # Depth settings
        'online_depth': 'DepthAnythingV2',  # or 'DepthAnything', None for precomputed
        'use_depth_loss': True,

        # Feature embeddings
        'load_embeddings': True,  # Set to False to disable (faster but less accurate)
        'embedding_dim': 64,  # Reduce from 512 for faster processing
        'online_emb': 'dino',  # Use online DINO features

        # Scene initialization
        'start_from_complete_pc': False,

        # FPS for visualization
        'fps': 30,
    },

    # ===========================
    # Gaussian Splatting Parameters
    # ===========================
    'scene': {
        # Gaussian properties
        'isotropic': False,
        'sh_degree': 3,  # Spherical harmonics degree (0-3)

        # Initialization
        'densify_every': 10,
        'prune_every': 10,
        'reset_opacities': False,
    },

    # ===========================
    # Camera Tracking
    # ===========================
    'tracking_cam': {
        'num_iters': 40,  # Number of iterations for camera tracking
        'lr': 0.01,
        'use_sil_for_loss': False,
        'sil_thres': 0.5,
        'use_l2_for_loss': False,
        'use_ssim_for_loss': False,
        'ignore_outlier_depth_loss': False,
        'use_uncertainty_for_loss_mask': False,
        'use_uncertainty_for_loss': False,
        'use_chamfer': False,
    },

    # ===========================
    # Object Tracking
    # ===========================
    'tracking_obj': {
        'num_iters_init': 500,  # Iterations for first frame
        'num_iters': 60,         # Iterations for subsequent frames
        'lr': 0.01,
        'use_sil_for_loss': False,
        'sil_thres': 0.5,
        'use_l2_for_loss': False,
        'use_ssim_for_loss': False,
        'ignore_outlier_depth_loss': False,
        'use_uncertainty_for_loss_mask': False,
        'use_uncertainty_for_loss': False,
    },

    # ===========================
    # Loss Weights
    # ===========================
    'loss_weights': {
        'im': 1.0,           # RGB loss
        'depth': 0.5,        # Depth loss
        'depth_l1': 0.5,     # Depth L1 loss
        'embedding': 0.2,    # Feature embedding loss (if using embeddings)
    },

    # ===========================
    # Gaussian Densification & Pruning
    # ===========================
    'add_gaussians': {
        'add_new_gaussians': True,
        'densify_dict': {
            'start_after': 0,
            'remove_big_after': 0,
            'stop_after': 9999999,
            'densify_every': 10,
            'grad_thresh': 0.0002,
            'min_opacity': 0.005,
            'percent_dense': 0.01,
            'split_big': True,
        },
        'sil_thres_gaussians': 0.5,
    },

    # ===========================
    # Point Tracking Configuration
    # ===========================
    'tracking': {
        # Number of query points (if not providing custom queries)
        'n_query_points': 1024,  # Uniform grid of sqrt(1024) x sqrt(1024) points

        # Tracking parameters
        'best_x': 1,  # Track best of x nearest Gaussians (1 = single best)
        'vis_thresh': 0.5,  # Visibility threshold
        'vis_thresh_start': 0.5,  # Visibility threshold for start frame
        'traj_len': 10,  # Length of trajectory trail in visualization

        # Grid visualization
        'visualize_grid': True,  # Whether to create dense grid visualization
        'grid_points': 2048,  # Number of points for grid visualization

        # Flow visualization
        'visualize_flow': False,  # Set to True to compute optical flow (slow)
    },

    # ===========================
    # Visualization
    # ===========================
    'viz': {
        'vis_all': True,  # Save all visualization outputs
        'vis_trajs': True,  # Visualize trajectories
        'vis_grid': False,  # Visualize grid (dense tracking)
        'vis_fg_only': False,  # Only visualize foreground points
    },

    # ===========================
    # Learning Rates (Advanced)
    # ===========================
    'lr': {
        'means3D': 0.00016,
        'rgb_colors': 0.0025,
        'unnorm_rotations': 0.001,
        'logit_opacities': 0.05,
        'log_scales': 0.001,
        'cam_unnorm_rots': 0.0001,
        'cam_trans': 0.0001,
        'embedding': 0.001,
    },

    # ===========================
    # Optimization Settings (Advanced)
    # ===========================
    'optimizer': {
        'type': 'Adam',  # Adam or SGD
        'beta1': 0.9,
        'beta2': 0.999,
        'eps': 1e-8,
    },

    # ===========================
    # Performance Settings
    # ===========================
    'performance': {
        # Memory optimization
        'gradient_checkpointing': False,  # Enable if OOM
        'mixed_precision': False,  # Enable for faster training (experimental)

        # Batch processing
        'process_batch_size': 1,  # Process N frames before optimizing (1 = online)
    },
}


# ===========================
# Presets for Different Use Cases
# ===========================

def get_fast_config():
    """Fast configuration for quick testing (lower quality)."""
    fast_config = config.copy()
    fast_config.update({
        'data': {
            **fast_config['data'],
            'desired_height': 270,
            'desired_width': 480,
            'load_embeddings': False,
            'every_x_frame': 2,  # Process every other frame
        },
        'tracking_obj': {
            **fast_config['tracking_obj'],
            'num_iters_init': 200,
            'num_iters': 30,
        },
        'tracking': {
            **fast_config['tracking'],
            'n_query_points': 256,
        },
    })
    return fast_config


def get_high_quality_config():
    """High quality configuration (slower, more accurate)."""
    hq_config = config.copy()
    hq_config.update({
        'data': {
            **hq_config['data'],
            'desired_height': 480,
            'desired_width': 640,
            'load_embeddings': True,
            'embedding_dim': 128,
        },
        'tracking_obj': {
            **hq_config['tracking_obj'],
            'num_iters_init': 1000,
            'num_iters': 100,
        },
        'tracking': {
            **hq_config['tracking'],
            'n_query_points': 4096,
            'best_x': 3,  # Use best of 3 for robustness
        },
    })
    return hq_config


# To use a preset, uncomment one of these:
# config = get_fast_config()
# config = get_high_quality_config()
