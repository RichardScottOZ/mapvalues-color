import rasterio
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage.morphology import opening, disk
from skimage.filters import sobel

# --- 1. Load and Preprocess ---
with rasterio.open("final_map.tif") as src:
    raster = src.read()  # Read all bands (RGB)
    profile = src.profile

# Reshape to pixels
pixels = raster[:3].reshape(3, -1).T  # (height * width, 3)

# --- 2. Advanced Noise Removal ---
def is_valid_pixel(pixel):
    r, g, b = pixel
    # Manual exclusions (black text, white borders, etc.)
    exclude_colors = [
        (0, 0, 0),       # Pure black
        (255, 255, 255), # Pure white
        (200, 200, 200), # Light gray (common in borders)
        (100, 100, 100)  # Dark gray (text shadows)
    ]
    if any(np.all(pixel == ec) for ec in exclude_colors):
        return False
    # Dynamic threshold for near-black
    if (r < 40 and g < 40 and b < 40):
        return False
    return True

valid_mask = np.array([is_valid_pixel(p) for p in pixels])
valid_pixels = pixels[valid_mask]

# --- 3. Edge Detection (for text removal) ---
edge_mask = sobel(np.mean(raster[:3], axis=0)) > 0.1
edge_mask = edge_mask.reshape(-1)
valid_pixels = valid_pixels[~edge_mask[valid_mask]]  # Remove edge pixels

# --- 4. Clustering ---
num_clusters = 6
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(valid_pixels)
dominant_colors = kmeans.cluster_centers_.astype(int)

# --- 5. Thickness Mapping ---
def luminance(color):
    return 0.299*color[0] + 0.587*color[1] + 0.114*color[2]

sorted_colors = sorted(dominant_colors, key=luminance)
thickness_values = np.linspace(5, 50, num_clusters)  # 5m to 50m
color_to_thickness = {tuple(color): thick for color, thick in zip(sorted_colors, thickness_values)}

# --- 6. Apply Classification ---
classified_raster = np.zeros(raster.shape[1:], dtype=np.float32)
for color, thickness in color_to_thickness.items():
    mask = (raster[0] == color[0]) & (raster[1] == color[1]) & (raster[2] == color[2])
    classified_raster[mask] = thickness

# --- 7. Morphological Cleaning ---
clean_mask = opening(classified_raster > 0, disk(2))  # Remove small noise
classified_raster[~clean_mask] = 0  # Set non-clean areas to 0

# --- 8. Save Result ---
profile.update(dtype="float32", count=1, nodata=0)
with rasterio.open("refined.tif", "w", **profile) as dst:
    dst.write(classified_raster, 1)

# --- Visualization ---
plt.figure(figsize=(12,4))
plt.subplot(131)
plt.imshow(raster[:3].transpose(1,2,0))
plt.title("Original Map")
plt.subplot(132)
plt.imshow([dominant_colors], aspect='auto')
plt.xticks([])
plt.title("Detected Colors")
plt.subplot(133)
plt.imshow(classified_raster, cmap='viridis')
plt.colorbar(label='Thickness (m)')
plt.title("Classified Output")
plt.tight_layout()
plt.show()
