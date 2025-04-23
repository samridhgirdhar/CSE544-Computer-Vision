#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import open3d as o3d
import numpy as np
from scipy.stats import ortho_group
import copy
import os
import glob
import re
import matplotlib.pyplot as plt

def find_available_pcds(dataset_path):
    """Find all available PCD files in the dataset folder and sort them."""
    pcd_files = glob.glob(os.path.join(dataset_path, "pointcloud_*.pcd"))
    
    
    def extract_number(filename):
        match = re.search(r'pointcloud_(\d+)\.pcd', filename)
        return int(match.group(1)) if match else -1
    
    pcd_files.sort(key=extract_number)
    
    if len(pcd_files) < 2:
        raise ValueError("Not enough PCD files found in the dataset folder. Need at least 2 files.")
    
    print(f"Found {len(pcd_files)} PCD files.")
    return pcd_files

def load_point_cloud(file_path):
    """Load a point cloud from a PCD file."""
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        if len(pcd.points) == 0:
            print(f"Warning: {file_path} loaded but contains 0 points")
            return None
        print(f"Loaded {file_path} with {len(pcd.points)} points")
        return pcd
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def find_consecutive_valid_pcds(pcd_files):
    """Find two consecutive valid PCD files."""
    for i in range(len(pcd_files) - 1):
        source = load_point_cloud(pcd_files[i])
        if source is None or len(source.points) == 0:
            continue
            
        target = load_point_cloud(pcd_files[i+1])
        if target is None or len(target.points) == 0:
            continue
            
        return source, target, pcd_files[i], pcd_files[i+1]
    
    raise ValueError("Could not find two consecutive valid PCD files.")

def preprocess_point_cloud(pcd, voxel_size=0.05):
    """Downsample and compute normals for a point cloud."""
    
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    
    return pcd_down

def prepare_dataset(source_pcd, target_pcd, voxel_size=0.05):
    """Prepare point cloud data for registration."""
    source_down = preprocess_point_cloud(source_pcd, voxel_size)
    target_down = preprocess_point_cloud(target_pcd, voxel_size)
    
    return source_down, target_down

def execute_point_to_point_icp(source, target, initial_transformation, threshold=0.05, max_iteration=100):
    """Execute point-to-point ICP registration."""
    
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(initial_transformation)
    
    
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source_transformed, target, threshold, np.identity(4))
    initial_fitness = evaluation.fitness
    initial_inlier_rmse = evaluation.inlier_rmse
    
    print(f"Initial fitness: {initial_fitness:.4f}, Initial RMSE: {initial_inlier_rmse:.4f}")
    
    
    result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )
    
    print(f"Estimated fitness: {result.fitness:.4f}, Estimated RMSE: {result.inlier_rmse:.4f}")
    
    return {
        'initial_fitness': initial_fitness,
        'initial_inlier_rmse': initial_inlier_rmse,
        'estimated_fitness': result.fitness,
        'estimated_inlier_rmse': result.inlier_rmse,
        'transformation_matrix': result.transformation
    }

def generate_initial_transformation():
    """Generate a valid initial transformation matrix with rotation and translation."""
    
    R = ortho_group.rvs(3)
    
    
    if np.linalg.det(R) < 0:
        R[:, 0] = -R[:, 0]  # Flip one column to ensure det = 1
    
    
    t = np.random.uniform(-0.5, 0.5, 3).reshape(3, 1)
    
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    
    return T

def save_point_cloud_visualization(source, target, transformation, output_file="registration_result.png"):
    """Save a visualization of point clouds to a file using matplotlib."""
    
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(transformation)
    
    
    source_points = np.asarray(source_transformed.points)
    target_points = np.asarray(target.points)
    
    
    max_points = 5000
    source_indices = np.random.choice(len(source_points), min(max_points, len(source_points)), replace=False)
    target_indices = np.random.choice(len(target_points), min(max_points, len(target_points)), replace=False)
    
    source_points = source_points[source_indices]
    target_points = target_points[target_indices]
    
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    
    ax.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], c='red', s=1, label='Source (Transformed)')
    ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c='blue', s=1, label='Target')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud Registration Result')
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([
        np.max([source_points[:, 0].max(), target_points[:, 0].max()]) - 
        np.min([source_points[:, 0].min(), target_points[:, 0].min()]),
        np.max([source_points[:, 1].max(), target_points[:, 1].max()]) - 
        np.min([source_points[:, 1].min(), target_points[:, 1].min()]),
        np.max([source_points[:, 2].max(), target_points[:, 2].max()]) - 
        np.min([source_points[:, 2].min(), target_points[:, 2].min()])
    ]).max() / 2.0
    
    mid_x = (source_points[:, 0].mean() + target_points[:, 0].mean()) / 2
    mid_y = (source_points[:, 1].mean() + target_points[:, 1].mean()) / 2
    mid_z = (source_points[:, 2].mean() + target_points[:, 2].mean()) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)
    
    print(f"Visualization saved to {output_file}")

def save_registration_metrics(result, source_file, target_file, output_file="registration_metrics.txt"):
    """Save the registration metrics to a text file."""
    with open(output_file, 'w') as f:
        f.write("Point Cloud Registration Results\n")
        f.write("===============================\n\n")
        f.write(f"Source file: {source_file}\n")
        f.write(f"Target file: {target_file}\n\n")
        
        f.write("Initial Transformation Matrix:\n")
        for row in result['initial_transformation']:
            f.write(f"{row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f} {row[3]:10.6f}\n")
        f.write("\n")
        
        f.write("Estimated Transformation Matrix:\n")
        for row in result['transformation_matrix']:
            f.write(f"{row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f} {row[3]:10.6f}\n")
        f.write("\n")
        
        f.write("Registration Metrics:\n")
        f.write(f"Initial fitness: {result['initial_fitness']:.6f}\n")
        f.write(f"Initial inlier RMSE: {result['initial_inlier_rmse']:.6f}\n")
        f.write(f"Estimated fitness: {result['estimated_fitness']:.6f}\n")
        f.write(f"Estimated inlier RMSE: {result['estimated_inlier_rmse']:.6f}\n")
    
    print(f"Registration metrics saved to {output_file}")

def main():
    
    dataset_path = "selected_pcds"  
    
    try:
        
        pcd_files = find_available_pcds(dataset_path)
        
        
        source, target, source_file, target_file = find_consecutive_valid_pcds(pcd_files)
        
        print(f"\nUsing files for registration:")
        print(f"Source: {source_file}")
        print(f"Target: {target_file}")
        
        
        source_down, target_down = prepare_dataset(source, target)
        
        
        initial_transformation = generate_initial_transformation()
        print("\nInitial transformation matrix:")
        print(initial_transformation)
        
        
        result = execute_point_to_point_icp(source_down, target_down, initial_transformation)
        
        
        result['initial_transformation'] = initial_transformation
        
        print("\nEstimated transformation matrix:")
        print(result['transformation_matrix'])
        
        
        print("\nSaving registration results...")
        save_point_cloud_visualization(source, target, result['transformation_matrix'])
        save_registration_metrics(result, source_file, target_file)
        
        
        print("\nRegistration Summary:")
        print(f"Initial fitness: {result['initial_fitness']:.4f}")
        print(f"Initial inlier RMSE: {result['initial_inlier_rmse']:.4f}")
        print(f"Estimated fitness: {result['estimated_fitness']:.4f}")
        print(f"Estimated inlier RMSE: {result['estimated_inlier_rmse']:.4f}")
        
        
        os.makedirs("results", exist_ok=True)
        
        
        source_transformed = copy.deepcopy(source)
        source_transformed.transform(result['transformation_matrix'])
        output_pcd_path = os.path.join("results", "transformed_source.pcd")
        o3d.io.write_point_cloud(output_pcd_path, source_transformed)
        print(f"Transformed source point cloud saved to {output_pcd_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


# In[ ]:


import open3d as o3d
import numpy as np
from scipy.stats import ortho_group
import copy
import os
import glob
import re
import matplotlib.pyplot as plt
import pandas as pd
import time

def find_available_pcds(dataset_path):
    """Find all available PCD files in the dataset folder and sort them."""
    pcd_files = glob.glob(os.path.join(dataset_path, "pointcloud_*.pcd"))
    
    
    def extract_number(filename):
        match = re.search(r'pointcloud_(\d+)\.pcd', filename)
        return int(match.group(1)) if match else -1
    
    pcd_files.sort(key=extract_number)
    
    if len(pcd_files) < 2:
        raise ValueError("Not enough PCD files found in the dataset folder. Need at least 2 files.")
    
    print(f"Found {len(pcd_files)} PCD files.")
    return pcd_files

def load_point_cloud(file_path):
    """Load a point cloud from a PCD file."""
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        if len(pcd.points) == 0:
            print(f"Warning: {file_path} loaded but contains 0 points")
            return None
        print(f"Loaded {file_path} with {len(pcd.points)} points")
        return pcd
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def find_consecutive_valid_pcds(pcd_files, index=0):
    """Find two consecutive valid PCD files starting from the given index."""
    if index >= len(pcd_files) - 1:
        raise ValueError("Index too large, not enough PCD files available.")
        
    source = load_point_cloud(pcd_files[index])
    if source is None or len(source.points) == 0:
        return find_consecutive_valid_pcds(pcd_files, index + 1)
        
    target = load_point_cloud(pcd_files[index + 1])
    if target is None or len(target.points) == 0:
        return find_consecutive_valid_pcds(pcd_files, index + 1)
        
    return source, target, pcd_files[index], pcd_files[index + 1]

def preprocess_point_cloud(pcd, voxel_size=0.05, compute_normals=True):
    """Downsample and compute normals for a point cloud."""
    
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    
    if compute_normals:
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    
    return pcd_down

def prepare_dataset(source_pcd, target_pcd, voxel_size=0.05, compute_normals=True):
    """Prepare point cloud data for registration."""
    source_down = preprocess_point_cloud(source_pcd, voxel_size, compute_normals)
    target_down = preprocess_point_cloud(target_pcd, voxel_size, compute_normals)
    
    return source_down, target_down

def generate_random_transformation():
    """Generate a valid initial transformation matrix with rotation and translation."""
    
    R = ortho_group.rvs(3)
    
    
    if np.linalg.det(R) < 0:
        
    
    
    t = np.random.uniform(-0.5, 0.5, 3).reshape(3, 1)
    
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    
    return T

def get_initial_transformation(method="random", source=None, target=None, voxel_size=0.05):
    """Get initial transformation based on specified method."""
    if method == "random":
        return generate_random_transformation()
    elif method == "identity":
        return np.eye(4)
    elif method == "ransac":
        # Use RANSAC for global registration
        if source is None or target is None:
            raise ValueError("Source and target point clouds are required for RANSAC initialization")
        
        # Prepare FPFH feature descriptors
        source_fpfh = compute_fpfh_features(source, voxel_size)
        target_fpfh = compute_fpfh_features(target, voxel_size)
        
        
        distance_threshold = voxel_size * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            4,  # Number of points to use for RANSAC
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
        )
        return result.transformation
    else:
        raise ValueError(f"Unknown initialization method: {method}")

def compute_fpfh_features(pcd, voxel_size):
    """Compute FPFH features for a point cloud."""
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5
    
    # Ensure normals are computed
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # Compute FPFH features
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    return fpfh

def execute_icp(source, target, initial_transformation, icp_method="point_to_point", 
                threshold=0.05, max_iteration=100, relative_fitness=1e-6, 
                relative_rmse=1e-6):
    """Execute ICP registration with specified parameters."""
    start_time = time.time()
    
    
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(initial_transformation)
    
    # Initial evaluation
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source_transformed, target, threshold, np.identity(4))
    initial_fitness = evaluation.fitness
    initial_inlier_rmse = evaluation.inlier_rmse
    
    # Set up ICP method
    if icp_method == "point_to_point":
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    elif icp_method == "point_to_plane":
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        raise ValueError(f"Unknown ICP method: {icp_method}")
    
    # Execute ICP
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iteration,
        relative_fitness=relative_fitness,
        relative_rmse=relative_rmse
    )
    
    result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, initial_transformation,
        estimation, criteria
    )
    
    end_time = time.time()
    
    
    diff_matrix = np.linalg.inv(initial_transformation) @ result.transformation
    
    R_diff = diff_matrix[:3, :3]
    
    rotation_error = np.linalg.norm(R_diff - np.eye(3), 'fro')
    
    translation_error = np.linalg.norm(diff_matrix[:3, 3])
    
    transformation_error = np.linalg.norm(diff_matrix - np.eye(4), 'fro')
    
    
    iterations_used = getattr(result, "iteration", max_iteration)
    
    return {
        'initial_fitness': initial_fitness,
        'initial_inlier_rmse': initial_inlier_rmse,
        'estimated_fitness': result.fitness,
        'estimated_inlier_rmse': result.inlier_rmse,
        'transformation_matrix': result.transformation,
        'transformation_error': transformation_error,
        'rotation_error': rotation_error,
        'translation_error': translation_error,
        'runtime': end_time - start_time,
        'iterations': iterations_used,
        'convergence': iterations_used < max_iteration
    }

def visualize_registration_result(source, target, transformation, output_file="registration_result.png"):
    """Visualize registration result and save to file."""
    
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(transformation)
    
    
    save_point_cloud_visualization(source, target, transformation, output_file)

def save_point_cloud_visualization(source, target, transformation, output_file="registration_result.png"):
    """Save a visualization of point clouds to a file using matplotlib."""
    
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(transformation)
    
    
    source_points = np.asarray(source_transformed.points)
    target_points = np.asarray(target.points)
    
    # Sample points if there are too many (for visualization performance)
    max_points = 5000
    source_indices = np.random.choice(len(source_points), min(max_points, len(source_points)), replace=False)
    target_indices = np.random.choice(len(target_points), min(max_points, len(target_points)), replace=False)
    
    source_points = source_points[source_indices]
    target_points = target_points[target_indices]
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], c='red', s=1, label='Source (Transformed)')
    ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c='blue', s=1, label='Target')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud Registration Result')
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([
        np.max([source_points[:, 0].max(), target_points[:, 0].max()]) - 
        np.min([source_points[:, 0].min(), target_points[:, 0].min()]),
        np.max([source_points[:, 1].max(), target_points[:, 1].max()]) - 
        np.min([source_points[:, 1].min(), target_points[:, 1].min()]),
        np.max([source_points[:, 2].max(), target_points[:, 2].max()]) - 
        np.min([source_points[:, 2].min(), target_points[:, 2].min()])
    ]).max() / 2.0
    
    mid_x = (source_points[:, 0].mean() + target_points[:, 0].mean()) / 2
    mid_y = (source_points[:, 1].mean() + target_points[:, 1].mean()) / 2
    mid_z = (source_points[:, 2].mean() + target_points[:, 2].mean()) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)
    
    print(f"Visualization saved to {output_file}")

def run_experiment(source, target, experiment_config, experiment_name):
    """Run a single experiment with the given configuration."""
    print(f"\nRunning experiment: {experiment_name}")
    print(f"Configuration: {experiment_config}")
    
    # Prepare data
    voxel_size = experiment_config.get('voxel_size', 0.05)
    compute_normals = experiment_config.get('icp_method', 'point_to_point') == 'point_to_plane'
    source_down, target_down = prepare_dataset(source, target, voxel_size, compute_normals)
    
    # Get initial transformation
    init_method = experiment_config.get('init_method', 'random')
    initial_transformation = get_initial_transformation(
        init_method, source_down, target_down, voxel_size
    )
    
    # Execute ICP
    result = execute_icp(
        source_down, 
        target_down, 
        initial_transformation,
        icp_method=experiment_config.get('icp_method', 'point_to_point'),
        threshold=experiment_config.get('threshold', 0.05),
        max_iteration=experiment_config.get('max_iteration', 100),
        relative_fitness=experiment_config.get('relative_fitness', 1e-6),
        relative_rmse=experiment_config.get('relative_rmse', 1e-6)
    )
    
    
    result['experiment_name'] = experiment_name
    result['initial_transformation'] = initial_transformation
    result['voxel_size'] = voxel_size
    result['icp_method'] = experiment_config.get('icp_method', 'point_to_point')
    result['threshold'] = experiment_config.get('threshold', 0.05)
    result['init_method'] = init_method
    
    # Save visualization for this experiment if requested
    if experiment_config.get('save_visualization', False):
        output_dir = "experiment_results"
        os.makedirs(output_dir, exist_ok=True)
        visualize_registration_result(
            source, 
            target, 
            result['transformation_matrix'],
            os.path.join(output_dir, f"{experiment_name}_result.png")
        )
    
    print(f"Experiment completed: {experiment_name}")
    print(f"Initial fitness: {result['initial_fitness']:.4f}, Initial RMSE: {result['initial_inlier_rmse']:.4f}")
    print(f"Estimated fitness: {result['estimated_fitness']:.4f}, Estimated RMSE: {result['estimated_inlier_rmse']:.4f}")
    print(f"Transformation error: {result['transformation_error']:.4f}")
    print(f"Runtime: {result['runtime']:.4f} seconds, Iterations: {result['iterations']}")
    
    return result

def format_transformation_matrix(matrix):
    """Format transformation matrix as a string for display."""
    result = ""
    for row in matrix:
        result += "[" + " ".join([f"{val:8.5f}" for val in row]) + "]\n"
    return result

def run_multiple_experiments(source, target, source_file, target_file):
    """Run multiple experiments with different hyperparameter settings."""
    # Define experiment configurations
    experiments = {
        "baseline": {
            "voxel_size": 0.05,
            "init_method": "random",
            "icp_method": "point_to_point",
            "threshold": 0.05,
            "max_iteration": 100,
            "save_visualization": True
        },
        "high_threshold": {
            "voxel_size": 0.05,
            "init_method": "random",
            "icp_method": "point_to_point",
            "threshold": 0.1,
            "max_iteration": 100
        },
        "low_threshold": {
            "voxel_size": 0.05,
            "init_method": "random",
            "icp_method": "point_to_point",
            "threshold": 0.02,
            "max_iteration": 100
        },
        "high_resolution": {
            "voxel_size": 0.02,
            "init_method": "random",
            "icp_method": "point_to_point",
            "threshold": 0.05,
            "max_iteration": 100
        },
        "low_resolution": {
            "voxel_size": 0.1,
            "init_method": "random",
            "icp_method": "point_to_point",
            "threshold": 0.05,
            "max_iteration": 100
        },
        "more_iterations": {
            "voxel_size": 0.05,
            "init_method": "random",
            "icp_method": "point_to_point",
            "threshold": 0.05,
            "max_iteration": 200
        },
        "point_to_plane": {
            "voxel_size": 0.05,
            "init_method": "random",
            "icp_method": "point_to_plane",
            "threshold": 0.05,
            "max_iteration": 100,
            "save_visualization": True
        },
        "identity_init": {
            "voxel_size": 0.05,
            "init_method": "identity",
            "icp_method": "point_to_point",
            "threshold": 0.05,
            "max_iteration": 100
        }
    }
    
    
    try:
        
        source_down, target_down = prepare_dataset(source, target, 0.05, True)
        _ = get_initial_transformation("ransac", source_down, target_down, 0.05)
        # If we get here, RANSAC works, so add the experiment
        experiments["ransac_init"] = {
            "voxel_size": 0.05,
            "init_method": "ransac",
            "icp_method": "point_to_point",
            "threshold": 0.05,
            "max_iteration": 100,
            "save_visualization": True
        }
        print("RANSAC initialization is available.")
    except Exception as e:
        print(f"RANSAC initialization is not available: {e}")
    
    # Run all experiments
    results = []
    for name, config in experiments.items():
        try:
            result = run_experiment(source, target, config, name)
            results.append(result)
        except Exception as e:
            print(f"Error in experiment {name}: {e}")
    
    # Create output directory
    output_dir = "experiment_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all results to CSV
    results_df = pd.DataFrame([
        {
            'Experiment': r['experiment_name'],
            'Init Method': r['init_method'],
            'ICP Method': r['icp_method'],
            'Voxel Size': r['voxel_size'],
            'Threshold': r['threshold'],
            'Initial Fitness': r['initial_fitness'],
            'Initial RMSE': r['initial_inlier_rmse'],
            'Final Fitness': r['estimated_fitness'],
            'Final RMSE': r['estimated_inlier_rmse'],
            'Transform Error': r['transformation_error'],
            'Rotation Error': r['rotation_error'],
            'Translation Error': r['translation_error'],
            'Iterations': r['iterations'],
            'Runtime (s)': r['runtime'],
            'Converged': r['convergence']
        }
        for r in results
    ])
    
    # Sort by Final RMSE (ascending)
    results_df = results_df.sort_values('Final RMSE')
    
    # Save detailed results and summary
    results_df.to_csv(os.path.join(output_dir, "experiment_results.csv"), index=False)
    
    # Generate summary report
    best_result = results_df.iloc[0]
    
    with open(os.path.join(output_dir, "experiment_summary.txt"), 'w') as f:
        f.write("Point Cloud Registration Experiments Summary\n")
        f.write("===========================================\n\n")
        f.write(f"Source file: {source_file}\n")
        f.write(f"Target file: {target_file}\n\n")
        
        f.write("Results Table (sorted by Final RMSE):\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("Best Experiment Configuration:\n")
        f.write(f"Experiment: {best_result['Experiment']}\n")
        f.write(f"Initialization Method: {best_result['Init Method']}\n")
        f.write(f"ICP Method: {best_result['ICP Method']}\n")
        f.write(f"Voxel Size: {best_result['Voxel Size']}\n")
        f.write(f"Threshold: {best_result['Threshold']}\n")
        f.write(f"Iterations: {best_result['Iterations']}\n\n")
        
        f.write("Best Experiment Results:\n")
        f.write(f"Initial Fitness: {best_result['Initial Fitness']:.6f}\n")
        f.write(f"Initial RMSE: {best_result['Initial RMSE']:.6f}\n")
        f.write(f"Final Fitness: {best_result['Final Fitness']:.6f}\n")
        f.write(f"Final RMSE: {best_result['Final RMSE']:.6f}\n")
        f.write(f"Transformation Error: {best_result['Transform Error']:.6f}\n")
        f.write(f"Runtime: {best_result['Runtime (s)']:.6f} seconds\n\n")
        
        # Get the best experiment's transformation matrix
        best_exp_result = next(r for r in results if r['experiment_name'] == best_result['Experiment'])
        
        f.write("Best Estimated Transformation Matrix:\n")
        f.write(format_transformation_matrix(best_exp_result['transformation_matrix']))
    
    print(f"\nAll experiment results saved to {output_dir}")
    print(f"Best experiment: {best_result['Experiment']} with RMSE: {best_result['Final RMSE']:.6f}")
    display(results_df)
    return results, results_df

def main():
    # Path to the dataset
    dataset_path = "selected_pcds"  # Update this with your actual path
    
    try:
        # Find all PCD files
        pcd_files = find_available_pcds(dataset_path)
        
        # Find two consecutive valid PCD files
        source, target, source_file, target_file = find_consecutive_valid_pcds(pcd_files)
        
        print(f"\nUsing files for registration:")
        print(f"Source: {source_file}")
        print(f"Target: {target_file}")
        
        # Run multiple experiments
        results, results_df = run_multiple_experiments(source, target, source_file, target_file)
        
        # Create visualizations for comparison
        output_dir = "experiment_results"
        
        # Create a bar chart for final RMSE
        plt.figure(figsize=(12, 6))
        results_df_plot = results_df.sort_values('Experiment')  # Sort for consistent ordering
        plt.bar(results_df_plot['Experiment'], results_df_plot['Final RMSE'])
        plt.title('Final RMSE by Experiment')
        plt.xlabel('Experiment')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rmse_comparison.png"), dpi=300)
        plt.close()
        
        # Create a bar chart for fitness
        plt.figure(figsize=(12, 6))
        plt.bar(results_df_plot['Experiment'], results_df_plot['Final Fitness'])
        plt.title('Final Fitness by Experiment')
        plt.xlabel('Experiment')
        plt.ylabel('Fitness')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fitness_comparison.png"), dpi=300)
        plt.close()
        
        # Plot initial vs final RMSE
        plt.figure(figsize=(12, 6))
        bar_width = 0.35
        r1 = np.arange(len(results_df_plot))
        r2 = [x + bar_width for x in r1]
        plt.bar(r1, results_df_plot['Initial RMSE'], width=bar_width, label='Initial RMSE')
        plt.bar(r2, results_df_plot['Final RMSE'], width=bar_width, label='Final RMSE')
        plt.xlabel('Experiment')
        plt.ylabel('RMSE')
        plt.title('Initial vs Final RMSE')
        plt.xticks([r + bar_width/2 for r in r1], results_df_plot['Experiment'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "initial_vs_final_rmse.png"), dpi=300)
        plt.close()
        
        print("\nAll visualizations created successfully!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


# In[5]:


import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import copy

def load_best_params():
    """Load the best parameters from experiment results."""
    try:
        # First try to load from the CSV file
        results_file = "experiment_results/experiment_results.csv"
        if os.path.exists(results_file):
            results_df = pd.read_csv(results_file)
            # Sort by Final RMSE (lower is better)
            results_df = results_df.sort_values('Final RMSE')
            best_exp = results_df.iloc[0]
            
            return {
                'experiment_name': best_exp['Experiment'],
                'init_method': best_exp['Init Method'],
                'icp_method': best_exp['ICP Method'],
                'voxel_size': best_exp['Voxel Size'],
                'threshold': best_exp['Threshold']
            }
        else:
            # If file doesn't exist, return default best params
            print("Experiment results file not found. Using default best parameters.")
            return {
                'experiment_name': 'default_best',
                'init_method': 'random',
                'icp_method': 'point_to_plane',
                'voxel_size': 0.05,
                'threshold': 0.05
            }
    except Exception as e:
        print(f"Error loading best parameters: {e}")
        # Return default values
        return {
            'experiment_name': 'default_best',
            'init_method': 'random',
            'icp_method': 'point_to_plane',
            'voxel_size': 0.05,
            'threshold': 0.05
        }

def load_point_cloud(file_path):
    """Load a point cloud from a PCD file."""
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        if len(pcd.points) == 0:
            print(f"Warning: {file_path} loaded but contains 0 points")
            return None
        print(f"Loaded {file_path} with {len(pcd.points)} points")
        return pcd
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def preprocess_point_cloud(pcd, voxel_size=0.05, compute_normals=True):
    """Downsample and compute normals for a point cloud."""
    # Downsample using voxel grid
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # Estimate normals if needed (for point-to-plane ICP)
    if compute_normals:
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    
    return pcd_down

def prepare_dataset(source_pcd, target_pcd, voxel_size=0.05, compute_normals=True):
    """Prepare point cloud data for registration."""
    source_down = preprocess_point_cloud(source_pcd, voxel_size, compute_normals)
    target_down = preprocess_point_cloud(target_pcd, voxel_size, compute_normals)
    
    return source_down, target_down

def get_initial_transformation(method="random", source=None, target=None, voxel_size=0.05):
    """Get initial transformation based on specified method."""
    from scipy.stats import ortho_group
    
    if method == "random":
        # Generate a random rotation matrix (3x3) from the orthogonal group
        R = ortho_group.rvs(3)
        
        # Ensure it's a proper rotation matrix (det = 1)
        if np.linalg.det(R) < 0:
            R[:, 0] = -R[:, 0]  # Flip one column to ensure det = 1
        
        # Generate a small random translation
        t = np.random.uniform(-0.5, 0.5, 3).reshape(3, 1)
        
        # Create the 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        
        return T
    elif method == "identity":
        return np.eye(4)
    elif method == "ransac":
        # Use RANSAC for global registration
        if source is None or target is None:
            raise ValueError("Source and target point clouds are required for RANSAC initialization")
        
        # Prepare FPFH feature descriptors
        source_fpfh = compute_fpfh_features(source, voxel_size)
        target_fpfh = compute_fpfh_features(target, voxel_size)
        
        # RANSAC registration
        distance_threshold = voxel_size * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            4,  # Number of points to use for RANSAC
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
        )
        return result.transformation
    else:
        raise ValueError(f"Unknown initialization method: {method}")

def compute_fpfh_features(pcd, voxel_size):
    """Compute FPFH features for a point cloud."""
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5
    
    # Ensure normals are computed
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # Compute FPFH features
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    return fpfh

def perform_best_registration(source, target, best_params):
    """Register source to target using the best parameters."""
    # Prepare datasets
    compute_normals = best_params['icp_method'] == 'point_to_plane'
    source_down, target_down = prepare_dataset(source, target, best_params['voxel_size'], compute_normals)
    
    # Get initial transformation
    initial_transformation = get_initial_transformation(
        best_params['init_method'], source_down, target_down, best_params['voxel_size']
    )
    
    # Set up ICP method
    if best_params['icp_method'] == 'point_to_point':
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    elif best_params['icp_method'] == 'point_to_plane':
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        raise ValueError(f"Unknown ICP method: {best_params['icp_method']}")
    
    # Execute ICP
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, best_params['threshold'], initial_transformation,
        estimation,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    
    print(f"ICP Registration Results:")
    print(f"Fitness: {result.fitness:.6f}")
    print(f"Inlier RMSE: {result.inlier_rmse:.6f}")
    
    return result.transformation

def visualize_transformed_cloud(source, target, transformation, output_dir="part3_results"):
    """Visualize and save the transformed point cloud compared to target."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Transform source
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(transformation)
    
    # Visualize original clouds
    fig = plt.figure(figsize=(12, 10))
    
    # Plot original source and target
    ax1 = fig.add_subplot(221, projection='3d')
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)
    
    # Sample points for better visualization
    max_points = 5000
    source_indices = np.random.choice(len(source_points), min(max_points, len(source_points)), replace=False)
    target_indices = np.random.choice(len(target_points), min(max_points, len(target_points)), replace=False)
    
    source_points = source_points[source_indices]
    target_points = target_points[target_indices]
    
    ax1.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], c='red', s=1, label='Source')
    ax1.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c='blue', s=1, label='Target')
    ax1.set_title('Original Point Clouds')
    ax1.legend()
    
    # Plot transformed source and target
    ax2 = fig.add_subplot(222, projection='3d')
    transformed_points = np.asarray(source_transformed.points)[source_indices]
    ax2.scatter(transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2], c='green', s=1, label='Transformed Source')
    ax2.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c='blue', s=1, label='Target')
    ax2.set_title('Transformed Source and Target')
    ax2.legend()
    
    # Plot side view
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], c='red', s=1, label='Source')
    ax3.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c='blue', s=1, label='Target')
    ax3.view_init(elev=0, azim=90)
    ax3.set_title('Side View - Original')
    
    # Plot side view of transformed
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.scatter(transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2], c='green', s=1, label='Transformed Source')
    ax4.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c='blue', s=1, label='Target')
    ax4.view_init(elev=0, azim=90)
    ax4.set_title('Side View - Transformed')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "transformed_visualization.png"), dpi=300)
    plt.close()
    
    # Create a colored point cloud visualization
    source_transformed.paint_uniform_color([1, 0, 0])  # Red
    target.paint_uniform_color([0, 0, 1])  # Blue
    
    # Get downsampled versions for cleaner visualization
    source_transformed_down = source_transformed.voxel_down_sample(0.05)
    target_down = target.voxel_down_sample(0.05)
    
    # Save to PCD files
    o3d.io.write_point_cloud(os.path.join(output_dir, "source_transformed.pcd"), source_transformed)
    o3d.io.write_point_cloud(os.path.join(output_dir, "target.pcd"), target)
    
    # Save the transformation matrix
    np.savetxt(os.path.join(output_dir, "transformation_matrix.txt"), transformation)
    
    # Generate error heatmap
    generate_error_heatmap(source_transformed, target, os.path.join(output_dir, "error_heatmap.png"))
    
    print(f"Visualization results saved to {output_dir}")
    
    return source_transformed

def generate_error_heatmap(source, target, output_file):
    """Generate a heatmap showing the error distance between source and target point clouds."""
    # Create a KD-tree from target points
    target_tree = o3d.geometry.KDTreeFlann(target)
    
    # Compute distances from each source point to the nearest target point
    source_points = np.asarray(source.points)
    distances = []
    
    # Only use a subset of points for large point clouds
    max_points = 10000
    if len(source_points) > max_points:
        indices = np.random.choice(len(source_points), max_points, replace=False)
        source_points = source_points[indices]
    
    for point in source_points:
        _, idx, dist = target_tree.search_knn_vector_3d(point, 1)
        distances.append(np.sqrt(dist[0]))
    
    # Create a scatter plot with distance-based coloring
    plt.figure(figsize=(10, 8))
    
    # Plot a 3D scatter with distances as colors
    ax = plt.axes(projection='3d')
    scatter = ax.scatter(
        source_points[:, 0], source_points[:, 1], source_points[:, 2],
        c=distances, 
        cmap='jet',
        s=1,
        vmin=0,
        vmax=max(0.1, np.percentile(distances, 95))  # Cap for better visualization
    )
    
    ax.set_title('Point Cloud Registration Error Heatmap')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add a color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Distance to nearest point (m)')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    # Calculate and print error statistics
    mean_error = np.mean(distances)
    median_error = np.median(distances)
    max_error = np.max(distances)
    std_error = np.std(distances)
    
    print(f"Error Statistics:")
    print(f"Mean error: {mean_error:.6f} m")
    print(f"Median error: {median_error:.6f} m")
    print(f"Max error: {max_error:.6f} m")
    print(f"Standard deviation: {std_error:.6f} m")
    
    # Save error statistics
    with open(output_file.replace(".png", "_stats.txt"), 'w') as f:
        f.write("Point Cloud Registration Error Statistics\n")
        f.write("=======================================\n\n")
        f.write(f"Mean error: {mean_error:.6f} m\n")
        f.write(f"Median error: {median_error:.6f} m\n")
        f.write(f"Max error: {max_error:.6f} m\n")
        f.write(f"Standard deviation: {std_error:.6f} m\n")

def analyze_transformation_matrix(transformation, output_file="part3_results/transformation_analysis.txt"):
    """Analyze the transformation matrix to understand what it does."""
    # Extract rotation matrix and translation vector
    rotation = transformation[:3, :3]
    translation = transformation[:3, 3]
    
    # Check if rotation matrix is valid
    det = np.linalg.det(rotation)
    is_orthogonal = np.allclose(np.dot(rotation, rotation.T), np.eye(3), atol=1e-6)
    
    # Compute rotation angle
    angle = np.arccos((np.trace(rotation) - 1) / 2)
    angle_degrees = np.degrees(angle)
    
    # Compute rotation axis
    if np.isclose(angle, 0) or np.isclose(angle, np.pi):
        # For 0 or 180 degrees, axis may not be well-defined
        rotation_axis = None
    else:
        # For other angles, compute axis as the eigenvector with eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        for i in range(len(eigenvalues)):
            if np.isclose(eigenvalues[i], 1.0):
                rotation_axis = np.real(eigenvectors[:, i])
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                break
    
    # Translation distance
    translation_distance = np.linalg.norm(translation)
    
    # Write analysis to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("Transformation Matrix Analysis\n")
        f.write("============================\n\n")
        
        f.write("Transformation Matrix:\n")
        for row in transformation:
            f.write(f"{row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f} {row[3]:10.6f}\n")
        f.write("\n")
        
        f.write(f"Rotation Matrix Determinant: {det:.6f}\n")
        f.write(f"Is Orthogonal: {is_orthogonal}\n\n")
        
        f.write(f"Rotation Angle: {angle_degrees:.2f} degrees\n")
        if rotation_axis is not None:
            f.write(f"Rotation Axis: [{rotation_axis[0]:.6f}, {rotation_axis[1]:.6f}, {rotation_axis[2]:.6f}]\n")
        else:
            f.write("Rotation Axis: Not well-defined (0 or 180 degrees rotation)\n")
        
        f.write(f"\nTranslation Vector: [{translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}]\n")
        f.write(f"Translation Distance: {translation_distance:.6f} meters\n")
        
        f.write("\nInterpretation:\n")
        f.write(f"The transformation represents a rotation of {angle_degrees:.2f} degrees ")
        if rotation_axis is not None:
            f.write(f"around the axis [{rotation_axis[0]:.4f}, {rotation_axis[1]:.4f}, {rotation_axis[2]:.4f}], ")
        f.write(f"followed by a translation of {translation_distance:.4f} meters in the direction ")
        f.write(f"[{translation[0]/translation_distance:.4f}, {translation[1]/translation_distance:.4f}, {translation[2]/translation_distance:.4f}].\n")
        
        f.write("\nThis represents the movement of the TurtleBot between the two consecutive scans.\n")
    
    print(f"Transformation analysis saved to {output_file}")
    
    # Print summary to console
    print("\nTransformation Summary:")
    print(f"Rotation: {angle_degrees:.2f} degrees")
    print(f"Translation: {translation_distance:.6f} meters")
    
    return {
        'rotation_angle': angle_degrees,
        'translation_distance': translation_distance,
        'rotation_axis': rotation_axis,
        'translation_vector': translation
    }

def main():
    """Main function for part 3."""
    # Set paths
    dataset_path = "selected_pcds"  # Update with your path
    source_file = "selected_pcds/pointcloud_0000.pcd"  # From previous output
    target_file = "selected_pcds/pointcloud_0004.pcd"  # From previous output
    output_dir = "part3_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load point clouds
    source = load_point_cloud(source_file)
    target = load_point_cloud(target_file)
    
    if source is None or target is None:
        print("Failed to load point clouds. Exiting.")
        return
    
    # Load best parameters
    best_params = load_best_params()
    print(f"Using best parameters: {best_params}")
    
    # Perform registration with best parameters
    transformation = perform_best_registration(source, target, best_params)
    
    # Visualize transformed point cloud
    source_transformed = visualize_transformed_cloud(source, target, transformation, output_dir)
    
    # Analyze transformation matrix
    transformation_analysis = analyze_transformation_matrix(transformation, os.path.join(output_dir, "transformation_analysis.txt"))


if __name__ == "__main__":
    main()


# In[8]:


import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
import pandas as pd
import copy
import time
from scipy.spatial.transform import Rotation as R

def find_all_point_clouds(dataset_path):
    """Find all PCD files in the dataset folder."""
    pcd_files = glob.glob(os.path.join(dataset_path, "pointcloud_*.pcd"))
    
    # Sort files by their index number
    def extract_number(filename):
        match = re.search(r'pointcloud_(\d+)\.pcd', filename)
        return int(match.group(1)) if match else -1
    
    pcd_files.sort(key=extract_number)
    
    if len(pcd_files) < 2:
        raise ValueError("Not enough PCD files found in the dataset folder. Need at least 2 files.")
    
    print(f"Found {len(pcd_files)} PCD files.")
    return pcd_files

def load_point_cloud(file_path):
    """Load a point cloud from a PCD file."""
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        if len(pcd.points) == 0:
            print(f"Warning: {file_path} loaded but contains 0 points")
            return None
        return pcd
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def preprocess_point_cloud(pcd, voxel_size=0.05, compute_normals=True):
    """Downsample and compute normals for a point cloud."""
    # Downsample using voxel grid
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # Estimate normals if needed (for point-to-plane ICP)
    if compute_normals:
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    
    return pcd_down

def register_point_clouds(source, target, voxel_size=0.05, icp_method="point_to_point", 
                         threshold=0.05, max_iteration=100):
    """Register source to target point cloud."""
    # Determine if normals are needed
    compute_normals = icp_method == "point_to_plane"
    
    # Preprocess point clouds
    source_down = preprocess_point_cloud(source, voxel_size, compute_normals)
    target_down = preprocess_point_cloud(target, voxel_size, compute_normals)
    
    # Get initial transformation (identity)
    initial_transformation = np.eye(4)
    
    # Set up ICP method
    if icp_method == "point_to_point":
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    elif icp_method == "point_to_plane":
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        raise ValueError(f"Unknown ICP method: {icp_method}")
    
    # Execute ICP
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, initial_transformation,
        estimation,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )
    
    return result

def register_sequential_point_clouds(pcd_files, voxel_size=0.05, icp_method="point_to_point", 
                                    threshold=0.05, max_iteration=100):
    """Register all point clouds sequentially."""
    # Load first point cloud
    pcds = []
    first_cloud = load_point_cloud(pcd_files[0])
    if first_cloud is None:
        raise ValueError(f"Could not load first point cloud: {pcd_files[0]}")
    
    pcds.append(first_cloud)
    
    # Global transformation list
    transformations = [np.eye(4)]  # First cloud has identity transformation
    
    # Position list for trajectory
    positions = [[0, 0, 0]]  # First position is origin
    
    # Results of pairwise registration
    pairwise_results = []
    
    # Log file for registration results
    os.makedirs("part4_results", exist_ok=True)
    with open("part4_results/registration_log.txt", "w") as log_file:
        log_file.write("Point Cloud Registration Log\n")
        log_file.write("==========================\n\n")
        
        # Register consecutive point clouds
        for i in range(1, len(pcd_files)):
            file_path = pcd_files[i]
            print(f"Processing {i}/{len(pcd_files)-1}: {os.path.basename(file_path)}")
            
            # Load current point cloud
            current_cloud = load_point_cloud(file_path)
            if current_cloud is None:
                print(f"Skipping {file_path} due to loading error")
                continue
            
            # Register with previous cloud
            registration_start = time.time()
            result = register_point_clouds(
                current_cloud, pcds[-1], 
                voxel_size, icp_method, 
                threshold, max_iteration
            )
            registration_time = time.time() - registration_start
            
            # Compute transformation relative to first cloud
            # Current = Previous * Transformation^(-1)
            # We invert because we register current to previous, but want to transform from first to current
            relative_transform = np.linalg.inv(result.transformation)
            global_transform = np.dot(transformations[-1], relative_transform)
            
            # Store results
            transformations.append(global_transform)
            
            # Extract position (translation part of the transformation)
            position = global_transform[:3, 3].tolist()
            positions.append(position)
            
            # Add transformed point cloud to list
            transformed_cloud = copy.deepcopy(current_cloud)
            transformed_cloud.transform(global_transform)
            pcds.append(transformed_cloud)
            
            # Store and log registration details
            pairwise_result = {
                'source': os.path.basename(file_path),
                'target': os.path.basename(pcd_files[i-1]),
                'fitness': result.fitness,
                'inlier_rmse': result.inlier_rmse,
                'registration_time': registration_time,
                'transformation': result.transformation,
                'global_transformation': global_transform,
                'position': position
            }
            pairwise_results.append(pairwise_result)
            
            # Write to log
            log_file.write(f"Pair {i-1} to {i}: {os.path.basename(pcd_files[i-1])} -> {os.path.basename(file_path)}\n")
            log_file.write(f"Fitness: {result.fitness:.6f}\n")
            log_file.write(f"Inlier RMSE: {result.inlier_rmse:.6f}\n")
            log_file.write(f"Registration time: {registration_time:.6f} seconds\n")
            log_file.write(f"Position: [{position[0]:.6f}, {position[1]:.6f}, {position[2]:.6f}]\n")
            log_file.write(f"Transformation Matrix:\n")
            for row in global_transform:
                log_file.write(f"{row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f} {row[3]:10.6f}\n")
            log_file.write("\n")
    
    # Save trajectory to CSV
    trajectory_df = pd.DataFrame(positions, columns=['x', 'y', 'z'])
    trajectory_df.to_csv("part4_results/trajectory.csv", index=False)
    
    return pcds, transformations, trajectory_df, pairwise_results

def visualize_registration_results(pcds, trajectory_df):
    """Visualize the registration results without requiring a display."""
    # Create output directory
    os.makedirs("part4_results", exist_ok=True)
    
    # 1. Visualize the trajectory in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory_df['x'], trajectory_df['y'], trajectory_df['z'], 'r-', linewidth=2)
    ax.scatter(trajectory_df['x'], trajectory_df['y'], trajectory_df['z'], c='blue')
    
    # Add point indices
    for i, (x, y, z) in enumerate(zip(trajectory_df['x'], trajectory_df['y'], trajectory_df['z'])):
        if i % 5 == 0 or i == len(trajectory_df) - 1:  # Label every 5th point and the last point
            ax.text(x, y, z, f'{i}', fontsize=9)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('TurtleBot 3D Trajectory')
    
    plt.tight_layout()
    plt.savefig("part4_results/trajectory_3d.png", dpi=300)
    plt.close()
    
    # 2. Visualize the trajectory in 2D (top-down view)
    plt.figure(figsize=(10, 8))
    plt.plot(trajectory_df['x'], trajectory_df['y'], 'r-', linewidth=2)
    plt.scatter(trajectory_df['x'], trajectory_df['y'], c='blue')
    
    # Add point indices
    for i, (x, y) in enumerate(zip(trajectory_df['x'], trajectory_df['y'])):
        if i % 5 == 0 or i == len(trajectory_df) - 1:  # Label every 5th point and the last point
            plt.text(x, y, f'{i}', fontsize=9)
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('TurtleBot 2D Trajectory (Top-Down View)')
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("part4_results/trajectory_2d.png", dpi=300)
    plt.close()
    
    # 3. Save the combined point cloud to file without visualizing
    combined_cloud = o3d.geometry.PointCloud()
    for pcd in pcds:
        combined_cloud += pcd
    
    # Save the combined point cloud to PCD file
    o3d.io.write_point_cloud("part4_results/combined_point_cloud.pcd", combined_cloud)
    
    # Create a simple visualization of the point cloud as colored points
    points = np.asarray(combined_cloud.points)
    
    # If the point cloud is too large, downsample for visualization
    if len(points) > 1000000:
        # Use voxel downsampling to reduce size
        combined_cloud = combined_cloud.voxel_down_sample(voxel_size=0.1)
        points = np.asarray(combined_cloud.points)
        print(f"Downsampled point cloud to {len(points)} points for visualization")
    
    # Create a scatter plot of points with fixed colors
    if len(points) > 0:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Use x coordinate for coloring (just for visualization)
        colors = points[:, 0]
        p = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=colors, cmap='viridis', s=0.5)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Combined Point Cloud')
        
        # Add a color bar
        fig.colorbar(p, ax=ax, label='X coordinate (m)')
        
        plt.tight_layout()
        plt.savefig("part4_results/combined_point_cloud.png", dpi=300)
        plt.close()
    
    print("Visualization results saved to part4_results/ directory")

def main():
    """Main function to run the point cloud registration pipeline."""
    # Configuration
    dataset_path = "selected_pcds"  # Adjust path as needed
    voxel_size = 0.05         # Adjust based on your best hyperparameters from part 2
    icp_method = "point_to_point"  # As required by the assignment
    threshold = 0.05          # Adjust based on your best hyperparameters from part 2
    max_iteration = 100       # Adjust based on your best hyperparameters from part 2
    
    # Find all point cloud files
    pcd_files = find_all_point_clouds(dataset_path)
    
    # Register all point clouds sequentially
    print("Registering point clouds...")
    pcds, transformations, trajectory_df, pairwise_results = register_sequential_point_clouds(
        pcd_files, voxel_size, icp_method, threshold, max_iteration
    )
    
    # Summarize registration results
    fitness_values = [result['fitness'] for result in pairwise_results]
    rmse_values = [result['inlier_rmse'] for result in pairwise_results]
    
    print("\nRegistration Summary:")
    print(f"Average Fitness: {np.mean(fitness_values):.6f}")
    print(f"Average Inlier RMSE: {np.mean(rmse_values):.6f}")
    print(f"Total Trajectory Length: {len(pcds)} points")
    
    # Create summary CSV
    summary_df = pd.DataFrame(pairwise_results)
    summary_df['source_index'] = range(1, len(pcd_files))
    summary_df['target_index'] = range(0, len(pcd_files)-1)
    
    # Select relevant columns for CSV
    summary_csv = summary_df[['source_index', 'target_index', 'fitness', 'inlier_rmse', 'registration_time']]
    summary_csv.to_csv("part4_results/registration_summary.csv", index=False)
    
    # Visualize results
    print("\nVisualizing results...")
    visualize_registration_results(pcds, trajectory_df)
    
    print("\nAll results saved to part4_results/ directory")

if __name__ == "__main__":
    main()

