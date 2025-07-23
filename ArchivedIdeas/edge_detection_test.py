import cv2

# Read the original image
img = cv2.imread('CS3/mosaic.jpg')
# Display original image
img = cv2.resize(img, (600,400))
cv2.imshow('Original', img)
cv2.waitKey(0)

# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)  # Canny Edge Detection
# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)

# Apply Laplacian operator
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# Convert to uint8
laplacian_abs = cv2.convertScaleAbs(laplacian)

# Display result
cv2.imshow("Laplacian Edge Detection", laplacian_abs)

cv2.waitKey(0)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    gray_vid = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Original', frame)
    edged_frame = cv2.Canny(frame, 100, 200)
    cv2.imshow('Edges', edged_frame)
    cv2.waitKey(0)

    # Apply Laplacian operator
    laplacian = cv2.Laplacian(frame, cv2.CV_64F)

    # Convert to uint8
    laplacian_abs = cv2.convertScaleAbs(laplacian)

    # Display result
    cv2.imshow("Laplacian Edge Detection", laplacian_abs)

    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()