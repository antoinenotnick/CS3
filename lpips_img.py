import lpips
import torchvision.transforms as transforms
from PIL import Image
from split_image import split_image
import shutil
import os
from abs_difference import abs_diff
from datetime import datetime
import csv

grid_rows = 16 # Default: 16
grid_cols = 16 # Default: 16
THRESHOLD = 0.1 # Default: 0.1
change_detected = False

before_file_name = 'RestoredSunBench'
after_file_name = 'MinorCurveDamageSunBench'

def load_image(path):
    img = Image.open(path).convert("RGB").resize((512, 512))
    transform = transforms.ToTensor()
    return transform(img).unsqueeze(0) * 2 - 1  # Normalize to [-1, 1]

def score(before_file_name='RestoredSunBench', after_file_name='OriginalSunBench', file_type='jpg', file='test'):
    # Deletes previous files stored in patches/
    if os.listdir('patches/'):
        for f in os.listdir('patches/'):
            if f.endswith(f'.{file_type}'):
                os.remove(os.path.join('patches/', f))

    split_image(f"images/{file}/{before_file_name}.{file_type}", grid_rows, grid_cols, False, False) # Adjust file format if needed
    split_image(f"images/{file}/{after_file_name}.{file_type}", grid_rows, grid_cols, False, False)

    global change_detected
    change_detected = False

    loss_fn = lpips.LPIPS(net='alex')  # 'alex', 'vgg', or 'squeeze'

    csv_path = "lpips_log.csv"
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(["Timestamp", "Patch Index", "LPIPS Score", "Before Image", "After Image"])

        for i in range(grid_rows * grid_cols):
            before_patch = f"patches/{before_file_name}_{i}.{file_type}"
            after_patch = f"patches/{after_file_name}_{i}.{file_type}"

            shutil.move(f"{before_file_name}_{i}.{file_type}", before_patch)
            shutil.move(f"{after_file_name}_{i}.{file_type}", after_patch)

            img_before = load_image(before_patch)
            img_after = load_image(after_patch)

            dist = loss_fn(img_before, img_after)
            dist_value = dist.item()

            if dist_value >= THRESHOLD:
                print(f"Patch {i} LPIPS Distance:", dist_value)
                change_detected = True
                csv_writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    i,
                    dist_value,
                    before_file_name,
                    after_file_name
                ])

            else:
                os.remove(before_patch)
                os.remove(after_patch)

    if change_detected:
        abs_diff(f"images/{file}/{before_file_name}.{file_type}", f"images/{file}/{after_file_name}.{file_type}")

if __name__ == "__main__":
    score(before_file_name=before_file_name.split('.')[0],
          after_file_name=after_file_name.split('.')[0],
          file_type='jpg',
          file='test')


"""
Plan:

# Find a way to integrate absolute difference based on reaching a particular Threshold
# Make the program work on different time intervals, constantly comparing the last image taken (you can prob use the computer camera just to test it)
! Detect when people come in front of the camera with a yolo model
! Optimize threshold and grid values (probably not)

Additional ideas: 
(Last priority) Solve Brightness Problem with abs_difference

# Log LPIPS scores to a CSV file.

Add YOLO object detection integration if people appear in the image.


- Done: move on to yolo prototype, gpt image analysis, and lidar when we get our hands on a camera

"""