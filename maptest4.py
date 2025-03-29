import rasterio
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from skimage.morphology import opening, disk
from skimage.color import rgb2lab, deltaE_cie76,lab2rgb  
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- 1. Enhanced Color Clustering ---
def cluster_colors(pixels, max_k=8):
    # Convert to CIELAB color space (better for color distance)
    lab_pixels = rgb2lab(pixels.reshape(-1, 1, 3)).reshape(-1, 3)
    
    # Find optimal clusters
    silhouette_scores = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(lab_pixels)
        if len(np.unique(labels)) > 1:
            silhouette_scores.append(silhouette_score(lab_pixels, labels))
        else:
            silhouette_scores.append(-1)
    
    optimal_k = np.argmax(silhouette_scores) + 2
    print(f"Detected {optimal_k} dominant color clusters")
    
    # Final clustering
    kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(lab_pixels)
    
    return kmeans.cluster_centers_, labels

# --- 2. Color Similarity Thresholding ---
def apply_threshold_classification(raster, cluster_centers, thickness_values, threshold=15):
    """Assign thickness based on color similarity within threshold"""
    classified = np.zeros(raster.shape[1:], dtype=np.float32)
    
    # Convert cluster centers to RGB
    rgb_centers = np.array([lab2rgb([[center]])[0,0] * 255 
                           for center in cluster_centers]).astype(int)
    
    # Process in batches to save memory
    for i in range(0, raster.shape[1], 100):
        for j in range(0, raster.shape[2], 100):
            batch = raster[:, i:i+100, j:j+100].transpose(1,2,0)
            batch_lab = rgb2lab(batch/255.)
            
            # Find closest cluster for each pixel
            distances = np.array([deltaE_cie76(batch_lab, center) 
                                for center in cluster_centers])
            closest = np.argmin(distances, axis=0)
            
            # Apply threshold
            min_dist = np.min(distances, axis=0)
            mask = min_dist <= threshold
            
            # Assign thickness
            for k, thick in enumerate(thickness_values):
                classified[i:i+100, j:j+100][(closest == k) & mask] = thick
                
    return classified

# --- 3. Full Pipeline ---
def process_map(input_path, output_path, min_thick=5, max_thick=50):
    # Load and preprocess
    with rasterio.open(input_path) as src:
        raster = src.read()
        profile = src.profile
    
    # Remove non-map pixels (same as before)
    pixels = raster[:3].reshape(3, -1).T
    valid_mask = ~np.all(pixels < [40,40,40], axis=1)  # Basic filter
    valid_pixels = pixels[valid_mask]
    
    # Cluster in LAB space
    lab_centers, _ = cluster_colors(valid_pixels)
    
    # Sort by luminance
    luminance = lab_centers[:,0]  # L* channel in LAB
    sorted_idx = np.argsort(luminance)
    thickness_values = np.linspace(min_thick, max_thick, len(lab_centers))
    
    # Classify with threshold
    classified = apply_threshold_classification(
        raster, 
        lab_centers[sorted_idx], 
        thickness_values,
        threshold=10  # Adjust based on color variation
    )
    
    # Clean small artifacts
    classified[~opening(classified > 0, disk(2))] = 0
    
    # Save result
    profile.update(dtype='float32', count=1, nodata=0)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(classified, 1)
    
    # Generate legend
    rgb_centers = np.array([lab2rgb([[center]])[0,0] * 255 
                          for center in lab_centers[sorted_idx]]).astype(int)
    legend_elements = [Patch(facecolor=color/255, 
                            label=f'{thick:.1f}m') 
                      for color, thick in zip(rgb_centers, thickness_values)]
    
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.imshow(classified, cmap='viridis')
    plt.colorbar(label='Thickness (m)')
    plt.subplot(122)
    plt.legend(handles=legend_elements, ncol=len(legend_elements))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path.replace('.tif','_legend.png'), dpi=300)
    
    return classified

# Run the processing
thickness_map = process_map("final_map.tif", "fuzzy.tif")
