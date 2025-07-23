import cv2
import numpy as np

img = cv2.imread("CS3/mosaic.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
tiles = []

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    area = cv2.contourArea(cnt)
    if len(approx) == 4 and area > 500:
        x, y, w, h = cv2.boundingRect(cnt)
        tiles.append((x, y, w, h))
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Now, build a grid and find missing spots

cv2.imshow("Detected Tiles", img)
cv2.waitKey(0)
cv2.destroyAllWindows()