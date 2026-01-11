import os
import shutil
from tqdm import tqdm
from PIL import Image
import glob
import os

def copy_images(src_path, dst_path):
    shutil.copy2(src_path, dst_path)

def resize_image_pil(image_path, new_width=1600):
    img = Image.open(image_path)
    if float(img.width) > 1600:
        w_percent = new_width / float(img.width)
        new_height = int((float(img.height) * w_percent))  # 计算等比例高度
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)  # 高质量降采样
        return img_resized, True
    else:
        return img, False

DatasetName = ["Caffe2"]
SourceImageDataset = r"/data2/xyw/Dataset/3DGS_Data/COLMAP"
On_The_Fly_Dataset = r"/data2/xyw/Dataset/3DGS_Data/On_the_Fly_SfM"

for dataset in DatasetName:
    SourceImagesDir = os.path.join(SourceImageDataset, r"{}/images".format(dataset))
    jpg_files = glob.glob(os.path.join(SourceImagesDir, "*.jpg"))
    endimage = jpg_files[0].split(".")[-1]
    IsResize = False

    if not os.path.exists(os.path.join(SourceImageDataset, r"{}/images_Resize".format(dataset))):
        progress_bar = tqdm(range(0, len(jpg_files)), desc="Resize progress {}".format(dataset))
        for jpg in jpg_files:
            img, IsResize = resize_image_pil(jpg)
            if IsResize:
                os.makedirs(SourceImagesDir + "_Resize", exist_ok=True)
                SavePath = os.path.join(SourceImageDataset, r"{}/images_Resize".format(dataset), jpg.split("/")[-1])
                # print(SavePath)
                img.save(SavePath)
            progress_bar.update(1)
        progress_bar.close()
    else:
        IsResize = True

    MainDir = os.path.join(On_The_Fly_Dataset, dataset)

    if IsResize:
        OriginSourceImagesDir = SourceImagesDir
        SourceImagesDir = os.path.join(SourceImageDataset, r"{}/images_Resize".format(dataset))

    print(MainDir)
    subfolders = [f for f in os.listdir(MainDir) if os.path.isdir(os.path.join(MainDir, f))]
    progress_bar = tqdm(range(0, len(subfolders)), desc="Copy progress {}".format(dataset))

    for i in range(len(subfolders)):
        if os.path.exists(os.path.join(MainDir, r"{}/sparse/0/imagesNames.txt".format(subfolders[i]))):
            ImagesNamesTXT = open(MainDir + r"/{}/sparse/0/imagesNames.txt".format(subfolders[i]))
            ImagesNamesList = ImagesNamesTXT.readline().split(",")
            os.makedirs(MainDir + r"/{}/images".format(subfolders[i]), exist_ok=True)
            for image in ImagesNamesList:
                if (not os.path.exists(MainDir + r"/{}/images/".format(subfolders[i]) + image + ".{}".format(endimage))) or True:
                    image = image.split("\n")[0]
                    copy_images(SourceImagesDir + "/" + image + ".{}".format(endimage), MainDir + r"/{}/images/".format(subfolders[i]) + image + ".{}".format(endimage))
        progress_bar.update(1)
    progress_bar.close()