# Define a brightness threshold (adjust as needed)
BRIGHTNESS_THRESHOLD = 40  # Values below this are considered "black writing"

# Calculate perceived brightness (luminance)
def luminance(color):
    r, g, b = color
    return 0.299 * r + 0.587 * g + 0.114 * b  # Human eye sensitivity formula

# Filter out black pixels before clustering
valid_pixels = [color for color in pixels if luminance(color) > BRIGHTNESS_THRESHOLD]
valid_pixels = np.array(valid_pixels)

# Ensure we have valid pixels left for clustering
if len(valid_pixels) == 0:
    raise ValueError("No valid colors found after filtering out black text. Try lowering the threshold.")

# Ensure we have valid pixels left for clustering
if len(valid_pixels) == 0:
    raise ValueError("No valid colors found after filtering out black text. Try lowering the threshold.")


# Run clustering on filtered pixels
num_clusters = 6  # Adjust as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(valid_pixels)

# Get dominant colors without black text
dominant_colors = kmeans.cluster_centers_.astype(int)

# Display detected colors
plt.figure(figsize=(8, 2))
plt.imshow([dominant_colors], aspect="auto")
plt.xticks(range(num_clusters), labels=[f"Color {i}" for i in range(num_clusters)])
plt.title("Detected Map Colors (Black Text Removed)")
plt.show()
