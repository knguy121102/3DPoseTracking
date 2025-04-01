import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class CameraParams:
    def __init__(self, R, t, K):
        """
        Initialize camera parameters
        
        Args:
            R: 3x3 rotation matrix
            t: 3x1 translation vector
            K: 3x3 camera intrinsic matrix
        """
        self.R = R
        self.t = t
        self.K = K
        
        # Calculate projection matrix P = K[R|t]
        self.P = np.zeros((3, 4))
        self.P[:3, :3] = R
        self.P[:3, 3] = t.flatten()
        self.P = K @ self.P

def triangulate_points(points1, points2, camera1, camera2):
    """
    Triangulate multiple 3D points from corresponding 2D points in different camera views
    
    Args:
        points1: Nx2 array of 2D points [x, y] in camera 1
        points2: Nx2 array of 2D points [x, y] in camera 2
        camera1: CameraParams for camera 1
        camera2: CameraParams for camera 2
        
    Returns:
        Nx3 array of 3D points [X, Y, Z]
    """
    # Reshape points for OpenCV
    points1 = np.array(points1, dtype=float).reshape(-1, 2)
    points2 = np.array(points2, dtype=float).reshape(-1, 2)
    
    # Triangulate using OpenCV
    points_4d = cv2.triangulatePoints(camera1.P, camera2.P, points1.T, points2.T)
    
    # Convert from homogeneous coordinates to 3D
    points_3d = points_4d[:3] / points_4d[3]
    
    return points_3d.T

def project_3d_to_2d(points_3d, camera):
    """
    Project multiple 3D points onto a camera's image plane
    
    Args:
        points_3d: Nx3 array of 3D points [X, Y, Z]
        camera: CameraParams 
        
    Returns:
        Nx2 array of 2D points [x, y]
    """
    # Convert to homogeneous coordinates
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    
    # Project using the camera's projection matrix
    points_2d_homogeneous = (camera.P @ points_3d_homogeneous.T).T
    
    # Convert back from homogeneous coordinates
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:3]
    
    return points_2d

def visualize(points1, points2, points_3d, camera1, camera2):
    """
    Visualize multiple 2D points and their triangulated 3D points
    
    Args:
        points1: Nx2 array of 2D points in camera 1
        points2: Nx2 array of 2D points in camera 2
        points_3d: Nx3 array of triangulated 3D points
        camera1: CameraParams for camera 1
        camera2: CameraParams for camera 2
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))
    
    # Plot camera 1 view (2D)
    ax1 = fig.add_subplot(131)
    ax1.set_title('Camera 1 View')
    ax1.set_xlim(0, 640)  # Assuming 640x480 image
    ax1.set_ylim(480, 0)  # Invert y-axis to match image coordinates
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.scatter(points1[:, 0], points1[:, 1], c='r', marker='o', s=50, label='Original Points')
    
    # Reproject 3D points to check accuracy
    reprojected_points1 = project_3d_to_2d(points_3d, camera1)
    ax1.scatter(reprojected_points1[:, 0], reprojected_points1[:, 1], c='g', marker='x', s=50, 
                label='Reprojected Points')
    ax1.legend()
    
    # Plot camera 2 view (2D)
    ax2 = fig.add_subplot(132)
    ax2.set_title('Camera 2 View')
    ax2.set_xlim(0, 640)  # Assuming 640x480 image
    ax2.set_ylim(480, 0)  # Invert y-axis to match image coordinates
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.scatter(points2[:, 0], points2[:, 1], c='r', marker='o', s=50, label='Original Points')
    
    # Reproject 3D points to check accuracy
    reprojected_points2 = project_3d_to_2d(points_3d, camera2)
    ax2.scatter(reprojected_points2[:, 0], reprojected_points2[:, 1], c='g', marker='x', s=50, 
                label='Reprojected Points')
    ax2.legend()
    
    # Plot 3D view
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title('3D View')
    ax3.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', marker='o', s=100, label='3D Points')
    
    # Plot camera positions
    # Camera 1
    cam1_pos = -np.linalg.inv(camera1.R) @ camera1.t
    ax3.scatter(cam1_pos[0], cam1_pos[1], cam1_pos[2], c='r', marker='^', s=100, label='Camera 1')
    
    # Camera 2
    cam2_pos = -np.linalg.inv(camera2.R) @ camera2.t
    ax3.scatter(cam2_pos[0], cam2_pos[1], cam2_pos[2], c='g', marker='^', s=100, label='Camera 2')
    
    # Plot lines from cameras to 3D points
    for i in range(len(points_3d)):
        ax3.plot([cam1_pos[0][0], points_3d[i, 0]], [cam1_pos[1][0], points_3d[i, 1]], 
                 [cam1_pos[2][0], points_3d[i, 2]], 'r--', alpha=0.2)
        ax3.plot([cam2_pos[0][0], points_3d[i, 0]], [cam2_pos[1][0], points_3d[i, 1]], 
                 [cam2_pos[2][0], points_3d[i, 2]], 'g--', alpha=0.2)
    
    # Plot cube edges
    # Define cube edges (pairs of point indices)
    edges = [
        (0,1), (1,2), (2,3), (3,0),  # Bottom face
        (4,5), (5,6), (6,7), (7,4),  # Top face
        (0,4), (1,5), (2,6), (3,7)   # Vertical edges
    ]
    
    for edge in edges:
        ax3.plot([points_3d[edge[0], 0], points_3d[edge[1], 0]],
                 [points_3d[edge[0], 1], points_3d[edge[1], 1]],
                 [points_3d[edge[0], 2], points_3d[edge[1], 2]], 'b-', alpha=0.5)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

def get_cube_points():
    """
    Generate 3D points for a cube and their 2D projections
    """
    # Define cube vertices (3D points)
    # Bottom face
    p0 = np.array([0, 0, 0])      # Front-left-bottom
    p1 = np.array([10, 0, 0])     # Front-right-bottom
    p2 = np.array([10, 10, 0])    # Back-right-bottom
    p3 = np.array([0, 10, 0])     # Back-left-bottom
    # Top face
    p4 = np.array([0, 0, 10])     # Front-left-top
    p5 = np.array([10, 0, 10])    # Front-right-top
    p6 = np.array([10, 10, 10])   # Back-right-top
    p7 = np.array([0, 10, 10])    # Back-left-top
    
    # Combine all points
    points_3d = np.array([p0, p1, p2, p3, p4, p5, p6, p7])
    
    return points_3d

def get_sphere_points(num_points=100):
    """
    Generate 3D points on a sphere
    
    Args:
        num_points: Number of points to generate on the sphere
        
    Returns:
        Nx3 array of 3D points [X, Y, Z] on the sphere
    """
    # Generate points using spherical coordinates
    phi = np.random.uniform(0, 2*np.pi, num_points)
    theta = np.random.uniform(0, np.pi, num_points)
    
    # Convert to Cartesian coordinates
    # Using radius of 5 units
    r = 5
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    # Center the sphere at (5, 5, 5)
    points_3d = np.column_stack((x, y, z)) + np.array([5, 5, 5])
    
    return points_3d

def visualize_sphere(points1, points2, points_3d, camera1, camera2):
    """
    Visualize sphere points in 2D and 3D
    
    Args:
        points1: Nx2 array of 2D points in camera 1
        points2: Nx2 array of 2D points in camera 2
        points_3d: Nx3 array of 3D points
        camera1: CameraParams for camera 1
        camera2: CameraParams for camera 2
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))
    
    # Plot camera 1 view (2D)
    ax1 = fig.add_subplot(131)
    ax1.set_title('Camera 1 View')
    ax1.set_xlim(0, 640)  # Assuming 640x480 image
    ax1.set_ylim(480, 0)  # Invert y-axis to match image coordinates
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.scatter(points1[:, 0], points1[:, 1], c='r', marker='o', s=20, label='Original Points')
    
    # Reproject 3D points to check accuracy
    reprojected_points1 = project_3d_to_2d(points_3d, camera1)
    ax1.scatter(reprojected_points1[:, 0], reprojected_points1[:, 1], c='g', marker='x', s=20, 
                label='Reprojected Points')
    ax1.legend()
    
    # Plot camera 2 view (2D)
    ax2 = fig.add_subplot(132)
    ax2.set_title('Camera 2 View')
    ax2.set_xlim(0, 640)  # Assuming 640x480 image
    ax2.set_ylim(480, 0)  # Invert y-axis to match image coordinates
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.scatter(points2[:, 0], points2[:, 1], c='r', marker='o', s=20, label='Original Points')
    
    # Reproject 3D points to check accuracy
    reprojected_points2 = project_3d_to_2d(points_3d, camera2)
    ax2.scatter(reprojected_points2[:, 0], reprojected_points2[:, 1], c='g', marker='x', s=20, 
                label='Reprojected Points')
    ax2.legend()
    
    # Plot 3D view
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title('3D View')
    ax3.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', marker='o', s=20, label='3D Points')
    
    # Plot camera positions
    # Camera 1
    cam1_pos = -np.linalg.inv(camera1.R) @ camera1.t
    ax3.scatter(cam1_pos[0], cam1_pos[1], cam1_pos[2], c='r', marker='^', s=100, label='Camera 1')
    
    # Camera 2
    cam2_pos = -np.linalg.inv(camera2.R) @ camera2.t
    ax3.scatter(cam2_pos[0], cam2_pos[1], cam2_pos[2], c='g', marker='^', s=100, label='Camera 2')
    
    # Plot lines from cameras to 3D points (with reduced opacity)
    for i in range(len(points_3d)):
        ax3.plot([cam1_pos[0][0], points_3d[i, 0]], [cam1_pos[1][0], points_3d[i, 1]], 
                 [cam1_pos[2][0], points_3d[i, 2]], 'r--', alpha=0.1)
        ax3.plot([cam2_pos[0][0], points_3d[i, 0]], [cam2_pos[1][0], points_3d[i, 1]], 
                 [cam2_pos[2][0], points_3d[i, 2]], 'g--', alpha=0.1)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Set up example camera parameters
    # In a real scenario, these would come from camera calibration
    
    # Camera 1 parameters (left camera)
    K1 = np.array([
        [100, 0, 320],  # Reduced focal length for better visibility
        [0, 100, 240],
        [0, 0, 1]
    ])
    R1 = np.eye(3)  # Identity rotation (camera aligned with world coordinates)
    t1 = np.array([[0, 0, 0]]).T  # Origin of world coordinates
    
    # Camera 2 parameters (right camera, 20 units to the right)
    K2 = np.array([
        [100, 0, 320],  # Same intrinsics as camera 1
        [0, 100, 240],
        [0, 0, 1]
    ])
    R2 = np.eye(3)  # Same orientation as camera 1
    t2 = np.array([[20, 0, 0]]).T  # 20 units to the right
    
    # Create camera parameter objects
    camera1 = CameraParams(R1, t1, K1)
    camera2 = CameraParams(R2, t2, K2)
    
    # First example: Cube
    print("\n=== Cube Example ===")
    points_3d_cube = get_cube_points()
    points1_cube = project_3d_to_2d(points_3d_cube, camera1)
    points2_cube = project_3d_to_2d(points_3d_cube, camera2)
    reconstructed_points_3d_cube = triangulate_points(points1_cube, points2_cube, camera1, camera2)
    visualize(points1_cube, points2_cube, reconstructed_points_3d_cube, camera1, camera2)
    
    # Second example: Sphere
    print("\n=== Sphere Example ===")
    points_3d_sphere = get_sphere_points(100)  # Generate 100 points on the sphere
    points1_sphere = project_3d_to_2d(points_3d_sphere, camera1)
    points2_sphere = project_3d_to_2d(points_3d_sphere, camera2)
    reconstructed_points_3d_sphere = triangulate_points(points1_sphere, points2_sphere, camera1, camera2)
    visualize_sphere(points1_sphere, points2_sphere, reconstructed_points_3d_sphere, camera1, camera2)
    
    # Print results for both examples
    print("\nCube Triangulation Results:")
    print("Original 3D Points:")
    print(points_3d_cube)
    print("\nReconstructed 3D Points:")
    print(reconstructed_points_3d_cube)
    
    print("\nSphere Triangulation Results:")
    print("Number of points:", len(points_3d_sphere))
    print("Original 3D Points (first 5):")
    print(points_3d_sphere[:5])
    print("\nReconstructed 3D Points (first 5):")
    print(reconstructed_points_3d_sphere[:5])

if __name__ == "__main__":
    main()