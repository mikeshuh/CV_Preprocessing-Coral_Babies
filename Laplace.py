import cv2 as cv

# Apply Laplace filter on 8-bit grayscale image
# https://docs.opencv.org/3.4/d5/db5/tutorial_laplace_operator.html
def laplace(gray_image, ksize=1):
    if ksize < 1 or ksize % 2 == 0:
        print("Error: Invalid ksize, must be positive and odd")
        return
    
    # Apply the Laplacian operator with a specified kernel size
    laplacian = cv.Laplacian(gray_image, cv.CV_64F, ksize=ksize)

    # Convert back to 8-bit to display
    laplacian = cv.convertScaleAbs(laplacian)

    return laplacian

def main():
    for i in range (1, 7):
        image = cv.imread(f"CoralBabies/{i}.JPG", cv.IMREAD_GRAYSCALE)

        laplace_image = laplace(image, 3)

        cv.imwrite(f"LaplaceOutput/laplace{i}.JPG", laplace_image)
    
if __name__ == "__main__":
    main()