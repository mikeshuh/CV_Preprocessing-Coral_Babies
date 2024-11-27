import cv2 as cv
import numpy as np

# Apply Hough Transform to detect lines in an edge-detected image
# https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html 
def hough_lines(edge_image, rho=1, theta=np.pi / 180, threshold=100):
    # Detect lines using Hough Line Transform
    lines = cv.HoughLines(edge_image, rho, theta, threshold)
    
    # Create an image to draw the lines on
    line_image = np.copy(edge_image) * 0  # Create a black image with the same dimensions
    
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
    return line_image

# Apply Hough Transform to detect circles in an edge-detected image
# https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html
def hough_circles(edge_image, dp=1.2, min_dist=20, param1=50, param2=30, min_radius=0, max_radius=0):
    # Detect circles using Hough Circle Transform
    circles = cv.HoughCircles(edge_image, cv.HOUGH_GRADIENT, dp, min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    
    # Create an image to draw the circles on
    circle_image = np.copy(edge_image) * 0  # Create a black image with the same dimensions
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the circle outline
            cv.circle(circle_image, (i[0], i[1]), i[2], (255, 255, 255), 2)
            # Draw the circle center
            cv.circle(circle_image, (i[0], i[1]), 2, (255, 255, 255), 3)
    
    return circle_image