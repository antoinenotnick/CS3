import cv2

# Load images
image1 = cv2.imread("CS3/images/BrokenMosaic.png")
image2 = cv2.imread("CS3/images/CompletelyFixedMosaic.png")

image1 = cv2.resize(image1, (600,400))
image2 = cv2.resize(image2, (600,400))

# Compute absolute difference
diff = cv2.absdiff(image1, image2)

# Convert to grayscale
gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# Threshold the difference
_, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding boxes around differences
for contour in contours:
    if cv2.contourArea(contour) > 100:  # Filter small noise
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 2) # Red rectangle

# Display results
cv2.imshow("Differences", image1)
cv2.waitKey(0)
cv2.destroyAllWindows()