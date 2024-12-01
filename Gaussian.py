import cv2 as cv

#Creates the filter of prescribed size and applies it to image
def gaussian_filter(image, kernel_size=15, sigma=10):
    return cv.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def main():
    for i in range (1, 7):
        image = cv.imread(f"CoralBabies/{i}.JPG")

        gaussian_image = gaussian_filter(image)

        cv.imwrite(f"GaussianOutput/gaussian{i}.JPG", gaussian_image)
    
if __name__ == "__main__":
    main()