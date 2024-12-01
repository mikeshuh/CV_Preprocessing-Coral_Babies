import cv2 as cv

#Turnes the input image into grayscale and applies sobel edge detection with kernel of chosen size
def sobel(image, kernel_size=3):
    sobel_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=kernel_size)
    return cv.magnitude(sobel_x, sobel_y)

def main():
    for i in range (1, 7):
        image = cv.imread(f"GaussianOutput/gaussian{i}.JPG", cv.IMREAD_GRAYSCALE)

        sobel_image = sobel(image)

        cv.imwrite(f"SobelOutput/sobel{i}.JPG", sobel_image)
    
if __name__ == "__main__":
    main()