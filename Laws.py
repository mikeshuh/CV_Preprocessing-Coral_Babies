import cv2 as cv
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
        filtered = cv.filter2D(image, -1, kernel)  # Convolve with the kernel
        energy = cv.GaussianBlur(filtered**2, (5, 5), 0)  # Smooth squared result
        energy_maps.append(energy)
    
    # Stack energy maps into a single feature vector
    feature_vector = np.stack([e.flatten() for e in energy_maps], axis=1)
    
    # Perform K-Means clustering with k-means++ initialization
    kmeans = KMeans(n_clusters=num_segments, init='k-means++', random_state=0).fit(feature_vector)
    labels = kmeans.labels_.reshape(image.shape)
    
    # Create a color-mapped segmentation image
    segmented_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
    # Define colors for segments
    colors_list = [
        [255, 0, 0],   # Red
        [0, 255, 0],   # Green
        [0, 0, 255],   # Blue
        # Add more colors if needed
    ]
    
    if num_segments <= len(colors_list):
        colors = np.array(colors_list[:num_segments])
    else:
        # If num_segments is greater than colors_list length, generate random colors
        colors = np.random.randint(0, 255, size=(num_segments, 3))
    
    for segment in range(num_segments):
        segmented_image[labels == segment] = colors[segment]
    
    return segmented_image

def main():
    for i in range(1, 7):
        image = cv.imread(f"CoralBabies/{i}.JPG", cv.IMREAD_GRAYSCALE)
        laws_segmented_image = laws_texture_segmentation(image)
        cv.imwrite(f"LawsOutput/laws{i}.JPG", laws_segmented_image)
    
if __name__ == "__main__":
    main()
