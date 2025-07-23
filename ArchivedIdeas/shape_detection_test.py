# Method 1
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# img = cv2.imread('CS3/shapes.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# # Process each contour
# for i, contour in enumerate(contours):
#     if i == 0:
#         continue

#     # Approximate contour shape
#     approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

#     # Draw contour
#     cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)

#     # Find center
#     M = cv2.moments(contour)
#     if M['m00'] != 0:
#         x = int(M['m10'] / M['m00'])
#         y = int(M['m01'] / M['m00'])

#     # Detect shape
#     sides = len(approx)
#     if sides == 3:
#         label = 'Triangle'
#     elif sides == 4:
#         label = 'Quadrilateral'
#     elif sides == 5:
#         label = 'Pentagon'
#     elif sides == 6:
#         label = 'Hexagon'
#     else:
#         label = 'Circle'

#     # Label the shape
#     cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
# cv2.imshow('shapes', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Method 2

# import numpy as np
# import cv2

# img = cv2.imread('CS3/shapes.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ret,thresh = cv2.threshold(gray,127,255,1)

# contours,h = cv2.findContours(thresh,1,2)

# for cnt in contours:
#     approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
#     print(len(approx))
#     if len(approx)==5:
#         print("pentagon")
#         cv2.drawContours(img,[cnt],0,255,-1)
#     elif len(approx)==3:
#         print("triangle")
#         cv2.drawContours(img,[cnt],0,(0,255,0),-1)
#     elif len(approx)==4:
#         print("square")
#         cv2.drawContours(img,[cnt],0,(0,0,255),-1)
#     elif len(approx) == 9:
#         print("half-circle")
#         cv2.drawContours(img,[cnt],0,(255,255,0),-1)
#     elif len(approx) > 15:
#         print("circle")
#         cv2.drawContours(img,[cnt],0,(0,255,255),-1)

# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Method 3

import numpy as np
import cv2
import random

#Reading the noisy image
img = cv2.imread("CS3/fuzzy.png",1)

#Displaying to see how it looks
cv2.imshow("Original",img)

#Converting the image to Gray Scale
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

#Removing Gaussian Noise
blur = cv2.GaussianBlur(gray, (3,3),0)

#Applying inverse binary due to white background and adapting thresholding for better results
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 205, 1)

#Checking to see how it looks
cv2.imshow("Binary",thresh)

#Finding contours with simple retrieval (no hierarchy) and simple/compressed end points
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#Checking to see how many contours were found
print(len(contours))

#An empty list to store filtered contours
filtered = []

#Looping over all found contours
for c in contours:
	#If it has significant area, add to list
	if cv2.contourArea(c) < 1000:continue
	filtered.append(c)

#Checking the number of filtered contours
print(len(filtered))

#Initialize an equally shaped image
objects = np.zeros([img.shape[0],img.shape[1],3], 'uint8')

#Looping over filtered contours
for c in filtered:
	#Select a random color to draw the contour
	col = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
	#Draw the contour on the image with above color
	cv2.drawContours(objects,[c], -1, col, -1)
	#Fetch contour area
	area = cv2.contourArea(c)
	#Fetch the perimeter
	p = cv2.arcLength(c,True)
	print(area,p)

#Finally show the processed image
cv2.imshow("Contours",objects)
	
#Closing protocol
cv2.waitKey(0)
cv2.destroyAllWindows()