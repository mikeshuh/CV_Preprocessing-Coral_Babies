import cv2 as cv

#Turnes the input image into graycale and applies canny dection with hysteresis parameters
def canny(image, candidateEdge=10, confirmedEdge=18):
    #https://www.geeksforgeeks.org/python-opencv-canny-function/
    canny = cv.Canny(image, candidateEdge, confirmedEdge)
    return canny

def main():
    for i in range (1, 7):
        image = cv.imread(f"Coral/GaussianOutput/gaussian{i}.JPG", cv.IMREAD_GRAYSCALE)

        canny_image = canny(image)

        cv.imwrite(f"Coral/CannyOutput/canny{i}.JPG", canny_image)
    
    for i in range (1, 3):
        image = cv.imread(f"Geology/GaussianOutput/gaussian{i}.JPG", cv.IMREAD_GRAYSCALE)

        canny_image = canny(image)

        cv.imwrite(f"Geology/CannyOutput/canny{i}.JPG", canny_image)

if __name__ == "__main__":
    main()