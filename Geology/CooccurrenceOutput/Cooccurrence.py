import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans

def normalization(image):
    height, width = image.shape  # Height and width of image
    image=image/64 #normalized


    for row in range(0, height):
        for column in range(0, width):
            image[row][column]=int(image[row][column]) #floor the values

    return image   # Normalize by 64

def cooccurrence_segmentation(image):
    normalized_image = normalization(image)  # Normalizes the image
    dictionary_list = []  # Creates list of dictionaries to cluster on
    neighborhood_size = 10  # Size of each covariance neighborhood
    height, width = image.shape  # Height and width of image
    
    
    for row in range(0, height, neighborhood_size):     #Iterate through dictionary
        for column in range(0, width, neighborhood_size):
            dictionary = {"00": 0, "01": 0, "02": 0, "03": 0,  #Cooccurrence Dictionary 
                          "10": 0, "11": 0, "12": 0, "13": 0,
                          "20": 0, "21": 0, "22": 0, "23": 0,
                          "30": 0, "31": 0, "32": 0, "33": 0}
            
            count=0
            for neighborhoodRow in range(0, neighborhood_size):    
                for neighborhoodColumn in range(0, neighborhood_size):
                    if (row+neighborhoodRow+1)<height and (column+neighborhoodColumn+1)<width:
                        temp = str(int(normalized_image[row+neighborhoodRow][column+neighborhoodColumn])) + str(int(normalized_image[row+neighborhoodRow+1][column+neighborhoodColumn+1]))
                        dictionary[temp] += 1
                        count+=1
                        
            for key in dictionary.keys(): #Average cooccurrence 
                if count!=0:
                    dictionary[key]/=count
            dictionary_list.append(dictionary)
               
    feature_vectors = [np.array(list(dictionary.values())) for dictionary in dictionary_list]  #Convert to array of np arrays
    
    # Set the number of clusters
    n_clusters = 3
    
    # Initialize KMeans and fit it to the feature vectors
    #https://scikit-learn.org/1.5/modules/generated/sklearn.cluster.KMeans.html
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(feature_vectors)
    
    # Assign each patch to a cluster
    labels = kmeans.labels_
    
    # Create color map (assign a unique color for each cluster)
    colors = [
        [255, 0, 0],   
        [0, 255, 0],   
        [0, 0, 255],   
    ]
    
    # Create an empty image with 3 channels (RGB)
    clustered_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Apply colors to the patches based on the cluster label
    neighborhood_index = 0
    for row in range(0, height, neighborhood_size):
        for column in range(0, width, neighborhood_size):
            cluster_label = labels[neighborhood_index] #Get patches label
            color = colors[cluster_label]  # Get the color for the cluster
            clustered_image[row:row + neighborhood_size, column:column + neighborhood_size] = color  # For visualization
            neighborhood_index += 1
    
    return clustered_image

def main():
    for i in range (1, 7):
        image = cv.imread(f"Coral/GaussianOutput/gaussian{i}.JPG", cv.IMREAD_GRAYSCALE)

        cooccurrence_image = cooccurrence_segmentation(image)

        cv.imwrite(f"Coral/CooccurrenceOutput/cooccurrence{i}.JPG", cooccurrence_image)
        print("partOne")

    for i in range (1, 3):
        image = cv.imread(f"Geology/GaussianOutput/gaussian{i}.JPG", cv.IMREAD_GRAYSCALE)

        cooccurrence_image = cooccurrence_segmentation(image)

        cv.imwrite(f"Geology/CooccurrenceOutput/cooccurrence{i}.JPG", cooccurrence_image)

    
if __name__ == "__main__":
    main()