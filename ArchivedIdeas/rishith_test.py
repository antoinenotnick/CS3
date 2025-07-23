import cv2
import pandas as pd

def detect_tile_damage(image_path, output_csv='damage_report.csv'):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found.")
        return


    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours (potential tiles)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    damage_report = []
    for idx, cnt in enumerate(contours):
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)


        # Filter small areas (noise)
        area = cv2.contourArea(cnt)
        if area < 100:  # threshold for tile size, adjust as needed
            continue


        # Get bounding box
        x, y, w, h = cv2.boundingRect(approx)


        # Check for missing/damaged tiles by shape irregularity
        shape = "Unknown"
        if len(approx) == 4:
            shape = "Quadrilateral"
        elif len(approx) > 4:
            shape = "Polygon"
        elif len(approx) < 4:
            shape = "Irregular/Damaged"


        # Check for damage: irregular shape or abnormal size
        damaged = False
        if shape == "Irregular/Damaged" or w < 10 or h < 10:
            damaged = True


        damage_report.append({
            'Tile Index': idx,
            'Location': f'({x},{y})',
            'Width': w,
            'Height': h,
            'Shape': shape,
            'Damaged': damaged
        })


        # Optionally, draw rectangles on damaged tiles
        if damaged:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)


    # Save annotated image
    cv2.imwrite('annotated_mosaic.png', img)


    # Output report as CSV
    df = pd.DataFrame(damage_report)
    df.to_csv(output_csv, index=False)
    print(f"Damage report saved to {output_csv}")
    print(df)


# Example usage:
detect_tile_damage('pool.png')