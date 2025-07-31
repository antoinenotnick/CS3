import cv2
import os
from datetime import datetime

def abs_diff(before_file_path, after_file_path, save_dir='images/diff_output'):
    # Load images
    image1 = cv2.imread(after_file_path)
    image2 = cv2.imread(before_file_path)

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
    bounding_box_count = 0
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter small noise
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 2) # Red rectangle
            bounding_box_count += 1

    # Save result
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"diff_{timestamp}.jpg")
    cv2.imwrite(save_path, image1)
    print(f"Saved diff image to: {save_path}")
    print(f"Number of differences detected: {bounding_box_count}")

    # Display results
    cv2.imshow("Differences", image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()