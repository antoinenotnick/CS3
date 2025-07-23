import lpips
import torchvision.transforms as transforms
from PIL import Image
from split_image import split_image
import shutil

grid_rows = 4
grid_cols = 4

before_file_name = "RestoredSunBench"
after_file_name = "OriginalSunBench"
file_type = "jpg"

split_image(f"images/{before_file_name}.{file_type}", grid_rows, grid_cols, False, False) # Adjust file format if needed
split_image(f"images/{after_file_name}.{file_type}", grid_rows, grid_cols, False, False)

def load_image(path):
    img = Image.open(path).convert("RGB").resize((256, 256))
    transform = transforms.ToTensor()
    return transform(img).unsqueeze(0) * 2 - 1  # Normalize to [-1, 1]

loss_fn = lpips.LPIPS(net='alex')  # 'alex', 'vgg', or 'squeeze'

for i in range(grid_rows * grid_cols):
    before_patch = f"patches/{before_file_name}_{i}.{file_type}"
    after_patch = f"patches/{after_file_name}_{i}.{file_type}"

    shutil.move(f"{before_file_name}_{i}.{file_type}", before_patch)
    shutil.move(f"{after_file_name}_{i}.{file_type}", after_patch)

    img_before = load_image(before_patch)
    img_after = load_image(after_patch)

    dist = loss_fn(img_before, img_after)
    print(f"Patch {i} LPIPS Distance:", dist.item())