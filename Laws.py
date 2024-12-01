import cv2
import numpy as np
from sklearn.cluster import KMeans

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

def laws_texture_segmentation(image, num_segments=3):
    # Generate Laws' kernels
    kernels = create_laws_kernels()
    
    # Compute texture energy
    energy_maps = []
    for kernel in kernels:
        filtered = cv2.filter2D(image, -1, kernel)  # Convolve with the kernel
        energy = cv2.GaussianBlur(filtered**2, (5, 5), 0)  # Smooth squared result
        energy_maps.append(energy)
    
    # Stack energy maps into a single feature vector
    feature_vector = np.stack([e.flatten() for e in energy_maps], axis=1)
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_segments, random_state=0).fit(feature_vector)
    labels = kmeans.labels_.reshape(image.shape)
    
    # Create a color-mapped segmentation image
    segmented_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    colors = np.random.randint(0, 255, size=(num_segments, 3))  # Random colors for each segment
    
    for segment in range(num_segments):
        segmented_image[labels == segment] = colors[segment]
    
    return segmented_image

# Load grayscale image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Normalize the image
image = image.astype(np.float32) / 255.0

# Apply Laws' texture segmentation
segmented_image = laws_texture_segmentation(image, num_segments=4)

# Save or display the result
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Alternatively, save the segmented image
cv2.imwrite('segmented_image.jpg', segmented_image)
