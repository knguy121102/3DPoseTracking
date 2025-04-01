# 3D Triangulation Visualization

This project demonstrates 3D point triangulation from multiple camera views using OpenCV and Python. It includes visualization tools for both 2D camera views and 3D reconstruction.

## Features

- Triangulation of 3D points from corresponding 2D points in different camera views
- Visualization of camera views and 3D reconstruction
- Example implementations for:
  - Cube reconstruction
  - Sphere reconstruction
- Interactive visualization using matplotlib

## Requirements

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

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script to see examples of cube and sphere triangulation:
```bash
python Triangulation.py
```

## Project Structure

- `Triangulation.py`: Main implementation file containing:
  - Camera parameter handling
  - 3D point triangulation
  - 2D projection
  - Visualization functions
  - Example implementations for cube and sphere

## License

MIT License 