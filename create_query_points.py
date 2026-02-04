"""
Interactive Query Point Selection Tool

This script allows you to interactively select query points on the first frame
of your sequence for tracking.

Usage:
    python create_query_points.py --sequence_dir /path/to/sequence --output queries.json

Controls:
    - Left click: Add query point
    - Right click: Remove nearest query point
    - 'r': Reset all points
    - 'q' or ESC: Quit and save
    - 'g': Generate grid of points
"""

import argparse
import os
import sys
import cv2
import numpy as np
import json
from pathlib import Path
import glob
from natsort import natsorted


class QueryPointSelector:
    """Interactive tool for selecting query points."""

    def __init__(self, image_path, grid_size=None):
        """
        Args:
            image_path: Path to the first frame image
            grid_size: If provided, generate NxN grid instead of manual selection
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")

        self.display_image = self.image.copy()
        self.points = []
        self.grid_size = grid_size

        # Window name
        self.window_name = "Query Point Selector"

        if grid_size:
            self.generate_grid(grid_size)
        else:
            print("\n" + "=" * 60)
            print("Interactive Query Point Selection")
            print("=" * 60)
            print("\nControls:")
            print("  Left click:  Add query point")
            print("  Right click: Remove nearest query point")
            print("  'r':         Reset all points")
            print("  'g':         Generate grid (will prompt for size)")
            print("  'q' or ESC:  Quit and save")
            print("\nClick on the image to add query points...")

    def generate_grid(self, n):
        """Generate NxN grid of query points."""
        h, w = self.image.shape[:2]

        # Create grid with margins
        margin = 20
        x_coords = np.linspace(margin, w - margin, n)
        y_coords = np.linspace(margin, h - margin, n)

        self.points = []
        for y in y_coords:
            for x in x_coords:
                self.points.append([int(x), int(y)])

        print(f"\nGenerated {len(self.points)} points in a {n}x{n} grid")
        self.update_display()

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point
            self.points.append([x, y])
            print(f"Added point #{len(self.points)} at ({x}, {y})")
            self.update_display()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove nearest point
            if len(self.points) > 0:
                # Find nearest point
                distances = [np.linalg.norm(np.array([x, y]) - np.array(p)) for p in self.points]
                nearest_idx = np.argmin(distances)

                if distances[nearest_idx] < 20:  # Within 20 pixels
                    removed_point = self.points.pop(nearest_idx)
                    print(f"Removed point at ({removed_point[0]}, {removed_point[1]})")
                    self.update_display()

    def update_display(self):
        """Update the display image with current points."""
        self.display_image = self.image.copy()

        # Draw all points
        for i, point in enumerate(self.points):
            # Draw circle
            cv2.circle(self.display_image, tuple(point), 5, (0, 255, 0), -1)
            # Draw point number
            cv2.putText(
                self.display_image,
                str(i + 1),
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1
            )

        # Draw stats
        stats_text = f"Points: {len(self.points)}"
        cv2.putText(
            self.display_image,
            stats_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow(self.window_name, self.display_image)

    def run(self):
        """Run the interactive selection."""
        if self.grid_size:
            # Just show grid
            self.update_display()
            print("\nPress any key to save and quit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return self.points

        # Interactive mode
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.update_display()

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('r'):  # Reset
                self.points = []
                print("\nReset all points")
                self.update_display()
            elif key == ord('g'):  # Generate grid
                try:
                    grid_size = int(input("\nEnter grid size (e.g., 32 for 32x32): "))
                    self.generate_grid(grid_size)
                except ValueError:
                    print("Invalid grid size")

        cv2.destroyAllWindows()
        return self.points

    def save_points(self, output_path):
        """Save query points to JSON file."""
        h, w = self.image.shape[:2]

        # Convert to normalized coordinates
        normalized_points = [[p[0] / w, p[1] / h] for p in self.points]

        data = {
            'points': self.points,  # Pixel coordinates
            'normalized_points': normalized_points,  # Normalized [0, 1]
            'image_size': {'width': w, 'height': h},
            'num_points': len(self.points),
            'source_image': self.image_path
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nSaved {len(self.points)} query points to {output_path}")


def find_first_frame(sequence_dir):
    """Find the first frame in a sequence directory."""
    # Try rgb folder first
    rgb_dir = os.path.join(sequence_dir, 'rgb')
    if os.path.exists(rgb_dir):
        images = natsorted(glob.glob(os.path.join(rgb_dir, '*.png')) +
                          glob.glob(os.path.join(rgb_dir, '*.jpg')))
        if images:
            return images[0]

    # Try JPEGImages folder (DAVIS format)
    jpeg_dir = os.path.join(sequence_dir, 'JPEGImages')
    if os.path.exists(jpeg_dir):
        images = natsorted(glob.glob(os.path.join(jpeg_dir, '*.jpg')) +
                          glob.glob(os.path.join(jpeg_dir, '*.png')))
        if images:
            return images[0]

    # Try root directory
    images = natsorted(glob.glob(os.path.join(sequence_dir, '*.png')) +
                      glob.glob(os.path.join(sequence_dir, '*.jpg')))
    if images:
        return images[0]

    raise ValueError(f"Could not find any images in {sequence_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive query point selection for tracking"
    )
    parser.add_argument(
        "--sequence_dir",
        type=str,
        required=True,
        help="Path to sequence directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="query_points.json",
        help="Output JSON file path (default: query_points.json)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Specific image path (if not provided, uses first frame from sequence)"
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=None,
        help="Generate NxN grid instead of interactive selection (e.g., --grid 32)"
    )

    args = parser.parse_args()

    # Find first frame
    if args.image is not None:
        image_path = args.image
    else:
        print(f"Looking for first frame in {args.sequence_dir}...")
        image_path = find_first_frame(args.sequence_dir)
        print(f"Using image: {image_path}")

    # Run selector
    selector = QueryPointSelector(image_path, grid_size=args.grid)
    points = selector.run()

    if len(points) == 0:
        print("\nNo points selected. Exiting without saving.")
        return

    # Save points
    selector.save_points(args.output)

    print("\n" + "=" * 60)
    print("Query Points Created Successfully!")
    print("=" * 60)
    print(f"\nYou can now use these points for tracking:")
    print(f"  python track_online_without_gt.py \\")
    print(f"    --config configs/custom/track_custom.py \\")
    print(f"    --sequence your_sequence \\")
    print(f"    --query_points {args.output}")


if __name__ == "__main__":
    main()
