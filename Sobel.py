import cv2 as cv

#Turnes the input image into grayscale and applies sobel edge detection with kernel of chosen size
def sobel(image, kernel_size=3):
    sobel_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=kernel_size)
    magnitude = cv.magnitude(sobel_x, sobel_y)
    return cv.convertScaleAbs(magnitude)

def main():
    for i in range (1, 7):
        image = cv.imread(f"Coral/GaussianOutput/gaussian{i}.JPG", cv.IMREAD_GRAYSCALE)

        sobel_image = sobel(image)

        cv.imwrite(f"Coral/SobelOutput/sobel{i}.JPG", sobel_image)
    
    for i in range (1, 3):
            image = cv.imread(f"Geology/GaussianOutput/gaussian{i}.JPG", cv.IMREAD_GRAYSCALE)

            sobel_image = sobel(image)

            cv.imwrite(f"Geology/SobelOutput/sobel{i}.JPG", sobel_image)

if __name__ == "__main__":
    main()