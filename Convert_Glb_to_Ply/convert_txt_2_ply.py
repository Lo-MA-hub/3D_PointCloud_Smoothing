import open3d as o3d
import numpy as np


def txt_to_ply_with_normals(txt_file: str, ply_file: str):
    """
    Convert a TXT file with x, y, z, r, g, b format into a PLY file with computed normals.

    :param txt_file: Path to the input TXT file.
    :param ply_file: Path to the output PLY file.
    """
    # Step 1: Load the TXT file
    print("Loading TXT file...")
    points = []
    colors = []
    with open(txt_file, 'r') as file:
        for line in file:
            x, y, z, r, g, b = map(float, line.strip().split())
            points.append([x, y, z])
            colors.append([r / 255.0, g / 255.0, b / 255.0])  # Normalize RGB to [0, 1]

    points = np.array(points)
    colors = np.array(colors)

    # Step 2: Create an Open3D PointCloud object
    print("Creating PointCloud...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Step 3: Estimate normals
    print("Computing normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=10)

    # Step 4: Save as PLY file
    print(f"Saving PLY file to {ply_file}...")
    o3d.io.write_point_cloud(ply_file, pcd, write_ascii=True)
    print("PLY file with normals saved successfully.")


# Example usage
txt_file = "/home/mingjun/PycharmProjects/smoothing/Floorplan_filling/Output_Data/office_walls_3m.txt"  # Replace with your TXT file path
ply_file = "/home/mingjun/PycharmProjects/smoothing/Floorplan_filling/Output_Data/office_walls_3m.ply"  # Output PLY file path
txt_to_ply_with_normals(txt_file, ply_file)
