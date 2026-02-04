"""
Configuration file for testing custom datasets without ground truth

This config demonstrates how to set up DynOMo for inference on new datasets
that don't have ground truth depth or annotations.

Usage:
    python test_without_gt.py --config configs/custom/test_custom.py --sequence my_sequence
"""

config = {
    'data': {
        'dataset_name': 'custom_test',  # Use the custom test dataset

        # Data paths
        'basedir': '/path/to/your/data',  # UPDATE THIS: Base directory containing sequences
        'sequence': 'test_sequence',  # Sequence name (will be overridden by --sequence arg)

        # Camera parameters
        # If your sequence has intrinsics.txt, these will be overridden
        # Otherwise, update these values to match your camera
        'desired_height': 480,
        'desired_width': 640,
        'camera_params': {
            'image_height': 480,
            'image_width': 640,
            'fx': 525.0,  # Focal length x (pixels)
            'fy': 525.0,  # Focal length y (pixels)
            'cx': 320.0,  # Principal point x (pixels)
            'cy': 240.0,  # Principal point y (pixels)
            'png_depth_scale': 1.0,  # Depth scale factor
        },

        # Frame sampling
        'every_x_frame': 1,  # Sample every x frames (1 = use all frames)
        'start': 0,  # Start frame index
        'end': -1,  # End frame index (-1 = all frames)

        # Depth prediction
        # Options: 'DepthAnything', 'DepthAnythingV2', or None (for precomputed depth)
        'online_depth': 'DepthAnythingV2',  # Use online depth prediction

        # Feature embeddings
        'load_embeddings': False,  # Set to True if you want to use feature embeddings
        'embedding_dim': 512,
        'online_emb': None,  # Options: 'dino', None

        # Visualization
        'fps': 30,  # Frames per second for output videos
    },

    # Model parameters (if running inference with a trained model)
    'model': {
        'device': 'cuda',  # 'cuda' or 'cpu'

        # Gaussian splatting parameters
        'num_gaussians': 100000,  # Number of Gaussians
        'isotropic': False,
        'sh_degree': 3,  # Spherical harmonics degree

        # Optimization parameters
        'lr': {
            'means3D': 0.00016,
            'rgb_colors': 0.0025,
            'unnorm_rotations': 0.001,
            'logit_opacities': 0.05,
            'log_scales': 0.001,
        },

        # Loss weights
        'loss_weights': {
            'im': 1.0,
            'depth': 0.1,
        },
    },

    # Visualization settings
    'viz': {
        'vis_all': True,  # Visualize all outputs
        'save_frames': True,  # Save individual frames
        'save_videos': True,  # Create videos from frames
    },
}
