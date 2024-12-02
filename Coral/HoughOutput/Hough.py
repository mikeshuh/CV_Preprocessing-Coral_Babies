import cv2 as cv
import numpy as np

# Apply Hough Transform to detect circles in an edge-detected image
# https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html
def hough_circles(edge_image, overlay_image, dp=2, min_dist=150, param1=50, param2=30, min_radius=25, max_radius=75):
    # Detect circles using Hough Circle Transform
    circles = cv.HoughCircles(edge_image, cv.HOUGH_GRADIENT, dp, min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the circle outline
            cv.circle(overlay_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the circle center
            cv.circle(overlay_image, (i[0], i[1]), 2, (0, 0, 255), 3)
    
    return overlay_image

def main():
    for i in range (1, 7):
        image = cv.imread(f"Coral/CoralBabies/{i}.JPG")
        edge_image = cv.imread(f"Coral/SobelOutput/sobel{i}.JPG", cv.IMREAD_GRAYSCALE)

        hough_circles_image = hough_circles(edge_image, image)

        cv.imwrite(f"Coral/HoughOutput/hough{i}.JPG", hough_circles_image)


    
if __name__ == "__main__":
    main()