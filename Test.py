import os
import shutil
SourceDirPath = r"/data2/xyw/Dataset/3DGS_Data/On_the_Fly_SfM/phantom3"
ToDirPath = r"/data2/xyw/Dataset/3DGS_Data/On_the_Fly_SfM_Origin/phantom3"

ToDirPath = os.path.join(ToDirPath, "GaussianSplatting")
os.makedirs(ToDirPath, exist_ok=True)

for i in range(400):
    if os.path.exists(os.path.join(SourceDirPath, str(i))):
        FromPath = os.path.join(SourceDirPath, str(i), "sparse", "0")
        ToPath = os.path.join(ToDirPath, str(i), "bin")
        os.makedirs(ToPath, exist_ok=True)

        src = os.path.join(FromPath, "cameras.bin")
        dst = os.path.join(ToPath, "cameras.bin")
        shutil.copy(src, dst)

        src = os.path.join(FromPath, "images.bin")
        dst = os.path.join(ToPath, "images.bin")
        shutil.copy(src, dst)

        src = os.path.join(FromPath, "points3D.bin")
        dst = os.path.join(ToPath, "points3D.bin")
        shutil.copy(src, dst)