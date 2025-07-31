import lpips
import cv2
import torchvision.transforms as transforms
from PIL import Image
from split_image import split_image
import shutil
import os
from abs_difference import abs_diff
import time
from datetime import datetime
import csv
import numpy as np
from skimage.exposure import match_histograms
from person_detector import PersonDetector

grid_rows = 8
grid_cols = 8
THRESHOLD = 0.25
change_detected = False
target_times = [
        "09:00",
        "10:00",
        "11:00",
        "12:00",
        "13:00",
        "14:00",
        "15:00",
        "16:00",
        "17:00"
    ] # Times when the camera takes a picture
# Default: Hourly from 09:00 to 17:00

person_detector = PersonDetector(confidence_threshold=0.5)

def load_image(path, reference_path=None):
    img = Image.open(path).convert("RGB").resize((512, 512))
    img_np = np.array(img)

    if reference_path:
        ref_img = Image.open(reference_path).convert("RGB").resize((512, 512))
        ref_np = np.array(ref_img)
        img_np = match_histograms(img_np, ref_np, channel_axis=-1)

    img_tensor = transforms.ToTensor()(Image.fromarray(img_np))
    
    return img_tensor.unsqueeze(0) * 2 - 1

def take_photo(base_filename='Captured', save_dir='images/camera', file_type='jpg'):
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"Directory '{save_dir}' does not exist.")

    # Get next available index
    existing_files = [
        f for f in os.listdir(save_dir)
        if f.startswith(base_filename) and f.endswith(f'.{file_type}')
    ]

    indices = [
        int(f.split('_')[-1].split('.')[0])
        for f in existing_files
        if f.split('_')[-1].split('.')[0].isdigit()
    ]

    next_index = max(indices, default=-1) + 1
    filename = f"{base_filename}_{next_index}.{file_type}"
    full_path = os.path.join(save_dir, filename)

    # Capture photo
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    ret, frame = cap.read()
    cap.release()

    if ret:
        cv2.imwrite(full_path, frame)
        print(f"Photo saved to {full_path}")
        return filename  # Return just the filename (not path)
    else:
        print("Error: Failed to capture image.")
        return None

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

    loss_fn = lpips.LPIPS(net='vgg')  # 'alex', 'vgg', or 'squeeze'

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
            img_after = load_image(before_patch, reference_path=after_patch)

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
    target_times_i = 0

    os.makedirs('images/camera', exist_ok=True)
    
    # Take reference image once
    reference_image = take_photo(base_filename='OriginalBench')

    while True:
        target_time = target_times[target_times_i]

        now = datetime.now().strftime("%H:%M")
        if now == target_time:

            # Check if person is in front of camera before taking photo
            if person_detector.capture_and_check_person():
                # Person detected - wait for clear view
                person_detector.wait_for_clear_view()

            current_image = take_photo(base_filename='Bench')
            if current_image:
                score(before_file_name=current_image.split('.')[0],
                      after_file_name=reference_image.split('.')[0],
                      file_type='jpg',
                      file='camera')
                
            target_times_i = (target_times_i + 1) % len(target_times)
            time.sleep(61)
        else:
            time.sleep(1)