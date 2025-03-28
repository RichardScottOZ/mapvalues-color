import rasterio
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Open the RGB raster
with rasterio.open("final_map.tif") as src:
    raster = src.read()  # Read all bands (RGB)
    profile = src.profile  # Store metadata

# Reshape raster to list of RGB triplets
pixels = raster[:3].reshape(3, -1).T  # (height * width, 3)

# Enhanced filtering to remove black text and other noise
def is_valid_pixel(pixel):
    r, g, b = pixel
    # Remove pure black (0,0,0) and near-black pixels
    if (r < 40 and g < 40 and b < 40):
        return False
    # Remove pure white (255,255,255) if present
    if (r > 250 and g > 250 and b > 250):
        return False
    return True

# Apply filtering
valid_mask = np.array([is_valid_pixel(p) for p in pixels])
valid_pixels = pixels[valid_mask]

# Verify we have enough valid pixels
if len(valid_pixels) == 0:
    raise ValueError("No valid pixels found after filtering. Adjust your thresholds.")

# Cluster colors using K-Means
num_clusters = 6  # Adjust based on map complexity
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(valid_pixels)

# Get unique cluster centers (dominant colors)
dominant_colors = kmeans.cluster_centers_.astype(int)

# Display the detected colors
plt.figure(figsize=(8, 2))
plt.imshow([dominant_colors], aspect="auto")
plt.xticks(range(num_clusters), labels=[f"Color {i}" for i in range(num_clusters)])
plt.title("Detected Map Colors (After Noise Removal)")
plt.show()

# Calculate brightness (luminance)
def luminance(color):
    r, g, b = color
    return 0.299 * r + 0.587 * g + 0.114 * b  # Human eye sensitivity formula

# Sort colors by brightness (low → high)
sorted_colors = sorted(dominant_colors, key=luminance)

# Assign thickness values
min_thickness, max_thickness = 5, 50  # Example: Min 5m, Max 50m cover
thickness_values = np.linspace(min_thickness, max_thickness, num_clusters)

# Create color-to-thickness mapping
color_to_thickness = {tuple(map(int, color)): thickness for color, thickness in zip(sorted_colors, thickness_values)}
print("Auto-generated color-to-thickness mapping:", color_to_thickness)

# Create an empty array for thickness values
classified_raster = np.zeros((raster.shape[1], raster.shape[2]), dtype=np.float32)

# Assign thickness based on detected colors
for (r, g, b), thickness in color_to_thickness.items():
    mask = (
        (raster[0] == r) & 
        (raster[1] == g) & 
        (raster[2] == b)
    )
    classified_raster[mask] = thickness

# Save the final raster with thickness values
profile.update(dtype="float32", count=1)  # Update metadata for single-band output
with rasterio.open("auto_clean.tif", "w", **profile) as dst:
    dst.write(classified_raster, 1)

print("Clean classified raster saved as auto_clean.tif")
