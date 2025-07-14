import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("CS3_Project\testimg.jpg", cv2.IMREAD_GRAYSCALE)
print(img)

# Apply Sobel operator
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges

# Compute gradient magnitude
gradient_magnitude = cv2.magnitude(sobelx, sobely)
 
# Convert to uint8
gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
 
# Display result
cv2.imshow("Sobel Edge Detection", gradient_magnitude)
 
cv2.waitKey(0)
cv2.destroyAllWindows()

# threshold = 6.

# img = cv2.imread('testimg.JPG')

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255

# sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
# sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
# sobel_xy = np.sqrt(sobelx ** 2 + sobely ** 2)
# edge_image = np.float32(sobel_xy > threshold)

# plt.imshow(edge_image, 'gray')
# plt.show()