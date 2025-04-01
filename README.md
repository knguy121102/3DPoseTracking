# 3D Triangulation Visualization

A Python implementation of 3D point triangulation from multiple camera views, featuring interactive visualization of both 2D camera views and 3D reconstruction.

## Overview

This project demonstrates how to reconstruct 3D points from corresponding 2D points in different camera views using OpenCV's triangulation algorithm. It includes visualization tools to help understand the triangulation process and verify the results.

## Features

- **3D Point Triangulation**: Reconstruct 3D points from corresponding 2D points in different camera views
- **Interactive Visualization**: 
  - 2D camera views showing original and reprojected points
  - 3D view showing reconstructed points and camera positions
  - Visual representation of triangulation rays
- **Example Implementations**:
  - Cube reconstruction (8 vertices)
  - Sphere reconstruction (100 points)
- **Camera Parameter Handling**: Support for camera intrinsics and extrinsics

## Dependencies

- Python 3.x
- OpenCV (cv2)
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/3d-triangulation.git
cd 3d-triangulation
```

2. Install the required packages:
```bash
pip install numpy opencv-python matplotlib
```

## Usage

Run the main script to see examples of cube and sphere triangulation:
```bash
python Triangulation.py
```

The script will:
1. Generate 3D points for a cube and sphere
2. Project these points onto two camera views
3. Reconstruct the 3D points using triangulation
4. Visualize the results in both 2D and 3D

## Project Structure

- `Triangulation.py`: Main implementation file containing:
  - `CameraParams` class: Handles camera parameters and projection matrices
  - `triangulate_points()`: Reconstructs 3D points from 2D correspondences
  - `project_3d_to_2d()`: Projects 3D points onto camera image planes
  - `visualize()`: Creates interactive visualizations
  - `get_cube_points()`: Generates cube vertices
  - `get_sphere_points()`: Generates points on a sphere

## Camera Setup

The example uses two cameras with the following parameters:
- Camera 1: Positioned at origin (0,0,0)
- Camera 2: Positioned 20 units to the right
- Both cameras have:
  - Focal length: 100
  - Principal point: (320, 240)
  - Resolution: 640x480

## Visualization

The visualization includes:
- Left panel: Camera 1 view with original and reprojected points
- Middle panel: Camera 2 view with original and reprojected points
- Right panel: 3D view showing:
  - Reconstructed points
  - Camera positions
  - Triangulation rays
  - Object structure (cube edges or sphere points)

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 