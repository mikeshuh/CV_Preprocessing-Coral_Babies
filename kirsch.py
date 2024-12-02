import cv2 as cv
import numpy as np

#Kirsh_Operator Edge Detection
def kirsch_compass_operator(image):
    # 8 Kirsch masks
    masks = [
        np.array([[5,  5,  5], [-3, 0, -3], [-3, -3, -3]]),  # North
        np.array([[-3, 5,  5], [-3, 0,  5], [-3, -3, -3]]),  # North East
        np.array([[-3, -3,  5], [-3, 0,  5], [-3, -3,  5]]), # East
        np.array([[-3, -3, -3], [-3, 0,  5], [-3,  5,  5]]), # South East
        np.array([[-3, -3, -3], [-3, 0, -3], [5,  5,  5]]),  # South
        np.array([[-3, -3, -3], [5, 0, -3], [5,  5, -3]]),   # South West
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),    # West
        np.array([[5,  5, -3], [5, 0, -3], [-3, -3, -3]])    # North West
    ]

    
    # Takes each operator and convolves it against the gray scale image
    #https://www.geeksforgeeks.org/python-opencv-filter2d-function/
    responses = [cv.filter2D(image, cv.CV_64F, operator) for operator in masks]

    #For each pixel it will take the maximum accross filters
    #https://realpython.com/numpy-max-maximum/
    max_response = np.max(responses, axis=0)  
    
    return max_response

def main():
    for i in range (1, 7):
        image = cv.imread(f"Coral/GaussianOutput/gaussian{i}.JPG", cv.IMREAD_GRAYSCALE)

        kirsch_image = kirsch_compass_operator(image)

        cv.imwrite(f"Coral/CreativeOutput/KirschOutput/kirsch{i}.JPG", kirsch_image)

    for i in range (1, 3):
        image = cv.imread(f"Geology/GaussianOutput/gaussian{i}.JPG", cv.IMREAD_GRAYSCALE)

        kirsch_image = kirsch_compass_operator(image)

        cv.imwrite(f"Geology/CreativeOutput/KirschOutput/kirsch{i}.JPG", kirsch_image)   
    
if __name__ == "__main__":
    main()