import cv2
import os
import numpy as np
from tqdm import tqdm

class self:
    def __init__(self):
        self.SaveDirRootPath = None
        self.Intermediate_RGB_Render_Path = None
        self.Intermediate_TDOM_Render_Path = None
        self.SaveImageNumber = 0
        self.SaveTDOMNumber = 0
        self.StartFromImageNo = 0

# 根据RGB影像的保存结果生成Demo
def GetRGBDemo(self):
    # 获取影像的尺寸
    first_image_path = os.path.join(self.Intermediate_RGB_Render_Path, "0.jpg")
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # 定义视频编码器和输出格式
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式，例如'XVID'或'mp4v'
    fps = 4  # 每秒帧数
    Demo_PreIMAGE = cv2.VideoWriter(os.path.join(self.SaveDirRootPath, "Render.mp4"), fourcc, fps, (width, height))

    # 将每帧添加到视频中
    progress_bar = tqdm(range(0, self.SaveImageNumber + self.StartFromImageNo), desc="Get_RGB_Demo", initial=0)
    for i in range(self.SaveImageNumber + self.StartFromImageNo):
        text = f"({i + 1}/{self.SaveImageNumber + self.StartFromImageNo})"

        if i <= self.StartFromImageNo - 1:
            black_frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(black_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            Demo_PreIMAGE.write(black_frame)
        else:
            img_path = os.path.join(self.Intermediate_RGB_Render_Path, f"{i - self.StartFromImageNo}.jpg")
            frame = cv2.imread(img_path)
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            Demo_PreIMAGE.write(frame)
        progress_bar.update(1)

    # 释放资源
    Demo_PreIMAGE.release()
    cv2.destroyAllWindows()

    print(f"RGB_Demo save at {os.path.join(self.SaveDirRootPath, 'Render.mp4')}")

# 根据TDOM的保存结果生成Demo
def GetTDOMDemo(self):
    # 获取影像的尺寸
    first_image_path = os.path.join(self.Intermediate_TDOM_Render_Path, "0.jpg")
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    scale = 0.5
    height = int(height * scale)
    width = int(width * scale)
    new_size = (width, height)

    # 定义视频编码器和输出格式
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式，例如'XVID'或'mp4v'
    fps = 4  # 每秒帧数
    Demo_TDOM = cv2.VideoWriter(os.path.join(self.args.Model_Path_Dir, "TDOM.mp4"), fourcc, fps, (width, height))

    # 将每帧添加到视频中
    progress_bar = tqdm(range(0, self.SaveTDOMNumber + self.StartFromImageNo), desc="Get_TDOM_Demo", initial=0)
    for i in range(self.SaveTDOMNumber + self.StartFromImageNo):
        text = f"TDOM: ({i + 1}/{self.SaveTDOMNumber + self.StartFromImageNo})"

        if i <= self.StartFromImageNo - 1:
            black_frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(black_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            Demo_TDOM.write(black_frame)
        else:
            img_path = os.path.join(self.Intermediate_TDOM_Render_Path, f"{i - self.StartFromImageNo}.jpg")
            frame = cv2.imread(img_path)
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            Demo_TDOM.write(frame)
        progress_bar.update(1)

    # 释放资源
    Demo_TDOM.release()
    cv2.destroyAllWindows()

    print(f"TDOM_Demo save at {os.path.join(self.args.Model_Path_Dir, 'TDOM.mp4')}")

self1 = self()
self1.SaveDirRootPath = r"/data2/xyw/Output/3DGS_Output/On_the_Fly_SfM/PolyTech_fine_V4/Intermediate_Results"
self1.Intermediate_RGB_Render_Path = os.path.join(self1.SaveDirRootPath, "Render")
self1.Intermediate_TDOM_Render_Path = os.path.join(self1.SaveDirRootPath, "TDOM")
self1.SaveImageNumber = 233
self1.SaveTDOMNumber = 0
self1.StartFromImageNo = 30

if self1.SaveImageNumber > 0:
    GetRGBDemo(self1)
if self1.SaveTDOMNumber > 0:
    GetTDOMDemo(self1)