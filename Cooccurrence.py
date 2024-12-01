import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans

def normalization(image):
    height, width = image.shape  # Height and width of image
    image=image/64 

    for row in range(0, height):
        for column in range(0, width):
            image[row][column]=int(image[row][column])

    return image   # Normalize by 64

def cooccurrence_segmentation(image):
    normalized_image = normalization(image)  # Normalizes the image
    dictionary_list = []  # Creates list of dictionaries to cluster on
    patch_size = 5  # Size of each covariance patch 
    height, width = image.shape  # Height and width of image
    
    
    for row in range(0, height, patch_size):
        for column in range(0, width, patch_size):
            dictionary = {"00": 0, "01": 0, "02": 0, "03": 0,
                          "10": 0, "11": 0, "12": 0, "13": 0,
                          "20": 0, "21": 0, "22": 0, "23": 0,
                          "30": 0, "31": 0, "32": 0, "33": 0}
            
            count=0
            for patchRow in range(0, patch_size):
                for patchColumn in range(0, patch_size):
                    if (row+patchRow+1)<height and (column+patchColumn+1)<width:
                        temp = str(int(normalized_image[row+patchRow][column+patchColumn])) + str(int(normalized_image[row+patchRow+1][column+patchColumn+1]))
                        dictionary[temp] += 1
                        count+=1
                        
            for key in dictionary.keys():
                if count!=0:
                    dictionary[key]/=count
            dictionary_list.append(dictionary)
               
    feature_vectors = [np.array(list(dictionary.values())) for dictionary in dictionary_list]
    
    # Set the number of clusters
    n_clusters = 3
    
    # Initialize KMeans and fit it to the feature vectors
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(feature_vectors)
    
    # Assign each patch to a cluster
    labels = kmeans.labels_
    
    # Create color map (assign a unique color for each cluster)
    colors = [
        [255, 0, 0],   # Red
        [0, 255, 0],   # Green
        [0, 0, 255],   # Blue
    ]
    
    # Create an empty image with 3 channels (RGB)
    clustered_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Apply colors to the patches based on the cluster label
    patch_index = 0
    for row in range(0, height, patch_size):
        for column in range(0, width, patch_size):
            cluster_label = labels[patch_index]
            color = colors[cluster_label]  # Get the color for the cluster
            clustered_image[row:row + patch_size, column:column + patch_size] = color  # For visualization
            patch_index += 1
    
    return clustered_image

def main():
    for i in range (1, 7):
        image = cv.imread(f"GaussianOutput/gaussian{i}.JPG", cv.IMREAD_GRAYSCALE)

        cooccurrence_image = cooccurrence_segmentation(image)

        cv.imwrite(f"CooccurrenceOutput/cooccurrence{i}.JPG", cooccurrence_image)
    
if __name__ == "__main__":
    main()