import lpips
import torch
import torchvision.transforms as transforms
from PIL import Image
from split_image import split_image
import shutil
import os
from abs_difference import abs_diff
from datetime import datetime
import csv

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]
GRID_SIZES = [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16)]

before_file_name = 'RestoredSunBench'
after_file_name = 'OriginalSunBench'
file_type = 'jpg'
file_folder = 'test'

# Clear old patches
for f in os.listdir('patches/'):
    if f.endswith(f'.{file_type}'):
        os.remove(os.path.join('patches', f))

def load_image(path):
    img = Image.open(path).convert("RGB").resize((512, 512))
    img_tensor = transforms.ToTensor()(img)  # [C, H, W]

    # Normalize brightness across all channels (convert to grayscale-like intensity)
    brightness = img_tensor.mean(dim=(1, 2), keepdim=True)  # Mean per channel
    img_tensor = img_tensor / (brightness + 1e-6)  # Prevent divide by zero

    # Clamp values to [0,1] after normalization
    img_tensor = torch.clamp(img_tensor, 0, 1)

    # Normalize to LPIPS expected range [-1, 1]
    return img_tensor.unsqueeze(0) * 2 - 1

def score(grid_rows, grid_cols, threshold):
    # Split images
    split_image(f"images/{file_folder}/{before_file_name}.{file_type}", grid_rows, grid_cols, False, False)
    split_image(f"images/{file_folder}/{after_file_name}.{file_type}", grid_rows, grid_cols, False, False)

    loss_fn = lpips.LPIPS(net='alex')  # 'alex', 'vgg', or 'squeeze'

    csv_path = "lpips_log.csv"
    file_exists = os.path.isfile(csv_path)
    change_detected = False

    with open(csv_path, mode='a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow([
                "Timestamp", "Patch Index", "LPIPS Score",
                "Before Image", "After Image",
                "Grid Rows", "Grid Cols", "Threshold"
            ])

        for i in range(grid_rows * grid_cols):
            before_patch = f"patches/{before_file_name}_{i}.{file_type}"
            after_patch = f"patches/{after_file_name}_{i}.{file_type}"

            shutil.move(f"{before_file_name}_{i}.{file_type}", before_patch)
            shutil.move(f"{after_file_name}_{i}.{file_type}", after_patch)

            img_before = load_image(before_patch)
            img_after = load_image(after_patch)

            dist = loss_fn(img_before, img_after)
            dist_value = dist.item()

            if dist_value >= threshold:
                print(f"[{grid_rows}x{grid_cols}] Patch {i} LPIPS: {dist_value:.4f}")
                change_detected = True
                csv_writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    i,
                    dist_value,
                    before_file_name,
                    after_file_name,
                    grid_rows,
                    grid_cols,
                    threshold
                ])
            else:
                os.remove(before_patch)
                os.remove(after_patch)

    if change_detected:
        abs_diff(
            f"images/{file_folder}/{before_file_name}.{file_type}",
            f"images/{file_folder}/{after_file_name}.{file_type}"
        )

if __name__ == "__main__":
    for threshold in THRESHOLDS:
        print(f"\n--- Running for Threshold: {threshold} ---")
        for rows, cols in GRID_SIZES:
            print(f"\n--- Running for Grid Size: {rows}x{cols} ---")
            score(grid_rows=rows, grid_cols=cols, threshold=threshold)

"""
Plan:

! Detect when people come in front of the camera with a yolo model
! Optimize threshold and grid values

Additional ideas: 
(Last priority) Solve Brightness Problem with abs_difference

Add YOLO object detection integration if people appear in the image.

- Done: move on to yolo prototype, gpt image analysis, and lidar when we get our hands on a camera

"""