# General Structure:
1. Make the original point cloud file smoother (make walls, ceiling, and floor smoother)
2. Fill the holes on the planes.
https://chatgpt.com/g/g-2DQzU5UZl-code-copilot/c/67539c81-dfd0-8009-835a-2025578c27a5
# For the smoothing part, we use Robust PCA based Algorithm 
the website link is: https://www.sciencedirect.com/science/article/pii/S1877050919307793
## Convert from glb to ply first.

## Algorithm Steps:
- Load the point cloud: parse the input .txt file to extract the 3D coordinates (x, y, z) and colors (r, g, b). 
- Estimate Normals using weighted PCA: use the k-nearest neighborhood (k-NN) to estimate normals robustly for each point using a weighted covariance matrix.
- Calculate Surface Variation Factor: the equation is as followed: 
___
![Equation](assets/dividing_different_regions_based_on_surface_variation_factor.png)
where $\lambda_0$, $\lambda_1$, $\lambda_2$ denote the three eigenvalues of the weighted covariance matrix.

- Classify Regions:
 - Compare the surface variation factor of each point with the average surface variation factor of its k-neighborhood.
 - Classify points into:
   - Flat regions: Small variation factor.
   -  Mutant regions: Large variation factor. 
- Apply Filters:
 - Flat: regions: Use improved Median Filtering to reduce noise.
 - Mutant regions: Use improved Bilateral Filtering to preserve edges and sharp features.
- Save Filtered Point Cloud:
 - Save the filtered data back into a .ply or .txt file.