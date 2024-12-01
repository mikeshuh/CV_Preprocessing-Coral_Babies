import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def create_laws_kernels():
    # Define 1D masks
    L5 = np.array([1, 4, 6, 4, 1])  # Level
    E5 = np.array([-1, -2, 0, 2, 1])  # Edge
    S5 = np.array([-1, 0, 2, 0, -1])  # Spot
    R5 = np.array([1, -4, 6, -4, 1])  # Ripple
    W5 = np.array([-1, 2, 0, -2, 1])  # Wave

    # Generate 2D kernels
    kernels = []
    masks = [L5, E5, S5, R5, W5]
    for i in masks:
        for j in masks:
            kernels.append(np.outer(i, j))  # Outer product to create 2D kernel
    return kernels

def laws_texture_energy(image, kernels):
    energy_maps = []
    for kernel in kernels:
        filtered = cv2.filter2D(image, -1, kernel)  # Convolve with the kernel
        energy = cv2.GaussianBlur(filtered**2, (5, 5), 0)  # Smooth squared result
        energy_maps.append(energy)
    return energy_maps

def segment_texture(energy_maps, num_segments=3):
    # Stack energy maps into a single feature vector
    feature_vector = np.stack([e.flatten() for e in energy_maps], axis=1)
    # K-Means clustering
    kmeans = KMeans(n_clusters=num_segments, random_state=0).fit(feature_vector)
    segmented = kmeans.labels_.reshape(energy_maps[0].shape)
    return segmented

# Load grayscale image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Normalize the image
image = image.astype(np.float32) / 255.0

# Generate Laws' kernels and compute texture energy
kernels = create_laws_kernels()
energy_maps = laws_texture_energy(image, kernels)

# Perform segmentation
segmented_image = segment_texture(energy_maps, num_segments=4)

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Segmented Image')
plt.imshow(segmented_image, cmap='jet')
plt.show()
