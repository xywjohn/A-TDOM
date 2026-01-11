import os
import shutil
from tqdm import tqdm
from PIL import Image
import glob

def resize_image_pil(image_path, new_width=1600):
    img = Image.open(image_path)
    if float(img.width) > 1600:
        w_percent = new_width / float(img.width)
        new_height = int((float(img.height) * w_percent))  # 计算等比例高度
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)  # 高质量降采样
        return img_resized, True
    else:
        return img, False

SourceImageDataset = r"D:\Study\3DGS\Dataset\COLMAP"
dataset = "npu"
SourceImagesDir = os.path.join(SourceImageDataset, r"{}\images".format(dataset))
jpg_files = glob.glob(os.path.join(SourceImagesDir, "*.jpg"))
endimage = jpg_files[0].split(".")[-1]
IsResize = False

for filename in os.listdir(SourceImagesDir):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.JPG')):
        continue

    img_path = os.path.join(SourceImagesDir, filename)
    img = Image.open(img_path)

    width, height = img.size

    # 如果是竖向（高>宽），则旋转90度
    if height > width:
        img = img.rotate(90, expand=True)

        # 保存到新路径
        img.save(os.path.join(SourceImagesDir, filename))

progress_bar = tqdm(range(0, len(jpg_files)), desc="Resize progress {}".format(dataset))
for jpg in jpg_files:
    img, IsResize = resize_image_pil(jpg)
    if IsResize:
        os.makedirs(SourceImagesDir + "_Resize", exist_ok=True)
        SavePath = os.path.join(SourceImageDataset, r"{}\images_Resize".format(dataset), jpg.split("\\")[-1])
        # print(SavePath)
        img.save(SavePath)
    progress_bar.update(1)
progress_bar.close()