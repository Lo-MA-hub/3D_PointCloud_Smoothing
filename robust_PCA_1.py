import open3d as o3d
import numpy as np
import cupy as cp  # GPU-accelerated array operations
from joblib import Parallel, delayed  # Parallel processing
from sklearn.neighbors import NearestNeighbors

def load_txt(file_path):
    """Load the TXT file with x y z r g b format."""
    data = np.loadtxt(file_path)
    points = data[:, :3]
    colors = data[:, 3:6] / 255.0  # Normalize RGB to [0, 1]
    return points, colors

def estimate_normals(points, k=15, n_jobs=-1):
    """Estimate normals using weighted PCA in parallel."""
    print("Estimating normals using parallel computation...")
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)

    def compute_normal(i):
        neighbors_idx = nbrs.kneighbors([points[i]], return_distance=False)[0]
        neighbors = points[neighbors_idx]
        cov_matrix = np.cov(neighbors - points[i], rowvar=False, bias=True)
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
        return eig_vecs[:, 0]  # Smallest eigenvector

    normals = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(compute_normal)(i) for i in range(len(points))
    )
    return np.array(normals)

def compute_surface_variation(points, normals, k=15, n_jobs=-1):
    """Compute surface variation factors in parallel."""
    print("Computing surface variation using parallel computation...")
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)

    def compute_variation(i):
        neighbors_idx = nbrs.kneighbors([points[i]], return_distance=False)[0]
        neighbors = points[neighbors_idx]
        cov_matrix = np.cov(neighbors - points[i], rowvar=False, bias=True)
        eig_vals = np.linalg.eigvalsh(cov_matrix)
        return eig_vals[0] / sum(eig_vals)

    variation = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(compute_variation)(i) for i in range(len(points))
    )
    return np.array(variation)
def filter_point_cloud(points, normals, variation, threshold, k=15):
    """Filter point cloud using improved median and bilateral filtering with GPU acceleration."""
    print("Applying filtering with GPU acceleration...")
    points_cp = cp.asarray(points)  # Move points to GPU
    normals_cp = cp.asarray(normals)
    variation_cp = cp.asarray(variation)

    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    filtered_points = cp.zeros_like(points_cp)  # Pre-allocate filtered points

    def process_point(i):
        neighbors_idx = nbrs.kneighbors([points[i]], return_distance=False)[0]
        neighbors = cp.asarray(points[neighbors_idx])  # Convert neighbors to CuPy array

        if variation_cp[i] < threshold:  # Flat region: Median filtering
            projections = cp.dot(neighbors - points_cp[i], normals_cp[i])
            median_proj = cp.median(projections)  # Use CuPy's median
            return points_cp[i] + median_proj * normals_cp[i]
        else:  # Mutant region: Bilateral filtering
            distances = cp.linalg.norm(neighbors - points_cp[i], axis=1)
            weights = cp.exp(-distances**2)
            filtered_proj = cp.sum(weights[:, None] * (neighbors - points_cp[i]), axis=0) / cp.sum(weights)
            return points_cp[i] + filtered_proj

    # Process each point and update the filtered_points array
    for i in range(len(points)):
        filtered_points[i] = process_point(i)

    return cp.asnumpy(filtered_points)  #


def save_ply(points, colors, output_file):
    """Save the point cloud as a PLY file."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_file, pcd)

def main(input_file, output_file):
    print("Loading point cloud...")
    points, colors = load_txt(input_file)

    print("Estimating normals...")
    normals = estimate_normals(points, k=15, n_jobs=64)

    print("Computing surface variation factors...")
    variation = compute_surface_variation(points, normals, k=15, n_jobs=64)

    print("Filtering point cloud...")
    threshold = np.mean(variation)  # Adaptive threshold
    filtered_points = filter_point_cloud(points, normals, variation, threshold, k=15)

    print("Saving filtered point cloud...")
    save_ply(filtered_points, colors, output_file)
    print(f"Filtered point cloud saved to {output_file}")

# Replace these paths with your actual file paths
input_file = "/home/mingjun/PycharmProjects/smoothing/Robust_Algorithm/Input_Data/glb_and_txt_file/segmented_ash.txt"
output_file = "/home/mingjun/PycharmProjects/smoothing/Robust_Algorithm/Output_Data/filtered_segmented_ash.ply"
main(input_file, output_file)
