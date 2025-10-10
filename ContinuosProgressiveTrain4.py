import os
import random
import torch
from random import randint
from utils.loss_utils import l1_loss
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from fused_ssim import fused_ssim
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from PIL import Image
import torchvision.transforms as transforms
from scipy.spatial import cKDTree
import math
import time
from scene.cameras import Camera, DummyCamera, DummyPipeline
from utils.camera_utils import camera_to_JSON
import json
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.general_utils import PILtoTorch
import numpy as np
import shutil
import glob
import cv2
import gc
import statistics
import copy
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import torchvision
import torch.multiprocessing as mp

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

WARNED = False

def prepare_output_and_logger(args, Create_for_new=False, Silence=False):
    # 创建对应的输出文件夹
    if not Create_for_new:
        m_path = args.model_path
    else:
        m_path = args.model_path_second

    if not Silence:
        print("Output folder: {}".format(m_path))
    os.makedirs(m_path, exist_ok=True)

    # 在文件夹中创建并打开一个二进制文件cfg_args，并在里面输出参数配置其的所有内容
    with open(os.path.join(m_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # 创建一个Tensorboard writer对象
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(m_path)
    else:
        if not Silence:
            print("Tensorboard not available: not logging progress")
    return tb_writer

def GetArgs():
    '''以下是在构建参数配置器'''
    # 构建一个空的原始参数配置器
    parser = ArgumentParser(description="Training script parameters")

    # 构建一个用于存储与模型有关参数的参数配置器
    lp = ModelParams(parser)

    # 构建一个用于存储与模型优化有关参数的参数配置器
    op = OptimizationParams(parser)

    # 构建一个用于存储与流程处理有关参数的参数配置器
    pp = PipelineParams(parser)

    # 为原始参数配置器加入一些参数
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)  # 是否开启某些异常检测，默认为False
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")  # quiet默认为False，当设置为True时，则在训练过程中不会输出任何内容到日志文件
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    '''以上是在构建参数配置器'''

    # 输出存储训练后模型的存储位置
    print("Optimizing " + args.Source_Path_Dir)

    return args, lp, op, pp

def TrainingPreparation(args):
    TimeCost = {}

    # 输出训练日志
    os.makedirs(args.Model_Path_Dir, exist_ok=True)

    if args.ContinueFromImageNo == -1:
        Diary = open(os.path.join(args.Model_Path_Dir, "Diary.txt"), "w")
        EvaluateDiary = open(os.path.join(args.Model_Path_Dir, "EvaluateDiary.txt"), "w")
        GaussianDiary = open(os.path.join(args.Model_Path_Dir, "GaussianDiary.txt"), "w")
    else:
        Diary = open(os.path.join(args.Model_Path_Dir, "Diary.txt"), "a")
        EvaluateDiary = open(os.path.join(args.Model_Path_Dir, "EvaluateDiary.txt"), "a")
        GaussianDiary = open(os.path.join(args.Model_Path_Dir, "GaussianDiary.txt"), "a")

    # 这个字典用于存储每一张影像已经被训练了多少次
    ImagesAlreadyBeTrainedIterations = {}

    # 一些输出上的设置
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # 读取所有的source_path
    IM = []
    for name in os.listdir(args.Source_Path_Dir):
        if os.path.isdir(os.path.join(args.Source_Path_Dir, name)) and int(name) >= args.StartFromImageNo:
            IM.append(int(name))

    # IM = [int(name) for name in os.listdir(args.Source_Path_Dir) if os.path.isdir(os.path.join(args.Source_Path_Dir, name))]
    IM = sorted(IM)
    print(IM)
    source_path_list = [os.path.join(args.Source_Path_Dir, str(im)) for im in IM]
    model_path_list = [os.path.join(args.Model_Path_Dir, str(im)) for im in IM]
    Diary.write(f"source_path_Dir: {args.Source_Path_Dir}\nModel_Path_Dir: {args.Model_Path_Dir}\n")
    Diary.write(f"Progress: {IM}\n")
    Diary.write("\n")

    Diary.write(f"Mean: {args.Mean}, "
                f"MergeScene_Densification_Interval: {args.MergeScene_Densification_Interval}, "
                f"OpacityThreshold: {args.opacity_threshold}, "
                f"InitialTrainingTimesSetZero: {args.InitialTrainingTimesSetZero}, "
                f"UseDifferentImageLr: {args.DifferentImagesLr}, "
                f"UseDepthLoss: {args.UseDepthLoss}, "
                f"UseScaleLoss: {args.UseScaleLoss}, "
                f"UseNormalLoss: {args.GetNormal}\n")
    print(f"Mean: {args.Mean}, "
          f"MergeScene_Densification_Interval: {args.MergeScene_Densification_Interval}, "
          f"OpacityThreshold: {args.opacity_threshold}, "
          f"InitialTrainingTimesSetZero: {args.InitialTrainingTimesSetZero}, "
          f"UseDifferentImageLr: {args.DifferentImagesLr}, "
          f"UseDepthLoss: {args.UseDepthLoss}, "
          f"UseScaleLoss: {args.UseScaleLoss}, "
          f"UseNormalLoss: {args.GetNormal}")

    # 如果是重新开始，则需要将数据库中输出的一些中间文件删除
    if args.ContinueFromImageNo == -1:
        for path in source_path_list:
            if os.path.exists(os.path.join(path, "args_output.txt")):
                os.remove(os.path.join(path, "args_output.txt"))
            if os.path.exists(os.path.join(path, "class_attributes.txt")):
                os.remove(os.path.join(path, "class_attributes.txt"))
            if os.path.exists(os.path.join(path, "sparse", "0", "points3D.ply")):
                os.remove(os.path.join(path, "sparse", "0", "points3D.ply"))

    # 如果使用原本的学习率更新方法，则重新设置position_lr_max_step
    if not args.DifferentImagesLr:
        WholeIterations = args.IterationFirstScene + (IM[-1] - IM[0] - 1) * args.IterationPerMergeScene + \
                          int((IM[-1] - IM[0] - 2) / args.GlobalOptimizationInterval) * args.GlobalOptimizationIteration + \
                          args.FinalOptimizationIterations
        args.position_lr_max_steps = WholeIterations

    return TimeCost, Diary, EvaluateDiary, GaussianDiary, ImagesAlreadyBeTrainedIterations, source_path_list, model_path_list, IM

def str2dict(str):
    replacestr = "{}\':,\n"
    for rstr in replacestr:
        str = str.replace(rstr, '')

    list = str.split(' ')
    dict = {}
    for i in range(int(len(list) / 2)):
        if '.' not in list[2 * i + 1]:
            dict[list[2 * i]] = int(list[2 * i + 1])
        else:
            dict[list[2 * i]] = float(list[2 * i + 1])
    return dict

def str2list(str):
    replacestr = "[]\',\n"
    for rstr in replacestr:
        str = str.replace(rstr, '')

    list = str.split(' ') if str != "" else []
    return list

def Get_LogImage(image, image_name="", OutputDirPath="", sigma=1.0, MaxGradLog=-1, save=False):
    R, G, B = image[0], image[1], image[2]
    gray = 0.2989 * R + 0.5870 * G + 0.1140 * B

    image = gray.squeeze(0).cpu().numpy()
    image = (image * 255).astype(np.uint8)

    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)

    log = cv2.Laplacian(blurred, cv2.CV_64F)
    log_abs = np.abs(log)
    # log_norm = log_abs
    if np.max(log_abs) >= MaxGradLog:
        log_norm = log_abs / np.max(log_abs)
        MG = np.max(log_abs)
    else:
        log_norm = log_abs / MaxGradLog
        MG = MaxGradLog

    psi = np.minimum(log_norm, 1.0)
    psi_inverted = psi

    if save and OutputDirPath != "" and image_name != "":
        os.makedirs(OutputDirPath, exist_ok=True)
        cv2.imwrite(os.path.join(OutputDirPath, f"{image_name}.jpg"), (psi_inverted * 255).astype(np.uint8))

    return psi, MG

def LogImage_minus(Log1, Log2, save_Dir, image_name, save=False):
    psi_B = np.maximum(Log1 - Log2, 0)

    if save:
        os.makedirs(save_Dir, exist_ok=True)

        save_path = os.path.join(save_Dir, image_name + ".jpg")
        cv2.imwrite(save_path, (psi_B * 255).astype(np.uint8))

    return psi_B

# 前端，第一个线程，主要用于读取数据以及一些数据预处理
class DataPreProccesser(mp.Process):
    # 构造函数
    def __init__(self):
        super().__init__()

        self.Silence = True

        # 用于两个进程之间的信息传递
        self.DataPreProccesser_queue = None
        self.GaussianTrainer_queue = None

        # 获取参数配置器
        self.args, self.lp, self.op, self.pp = GetArgs()

        # 用于统计峰值内存以及训练时间
        self.Peak_Memory = 0
        self.PhaseStartTime = None
        self.PhaseEndTime = None

        # 训练开始前的准备工作
        self.TimeCost, self.Diary, self.EvaluateDiary, self.GaussianDiary, self.ImagesAlreadyBeTrainedIterations, self.source_path_list, self.model_path_list, self.IM = TrainingPreparation(self.args)

        # 用于在渐进式训练时记录此次加入的影像
        self.NewCams = []

        # 记录模型的已训练次数
        self.AlreadyTrainingIterations = 0
        self.StartFromTrainingIterations = 0

        # 在没有开始渐进式训练之前，影像匹配矩阵和影像权重不存在
        self.ImageMatchingMatrix = None
        self.Images_Weights = None

        # 在没有开始渐进式训练之前，新增影像数量不存在
        self.NewImagesNum = -1

        # 该字典用于存储每一张影像对应的高斯球
        self.Image_Visibility = {}
        self.MaxWeightsImages = []

        # 该列表用于存储一些影像的名称，仅在需要进行分块训练时会被用到
        # 若该列表不为空，则训练影像时只会从该列表中的影像来选取
        self.ChosenImageNames = []

        self.targetcam = None
        self.TDOM_Cam = None
        self.TDOM_pipeline = None
        self.targetcam_savetimes = 0
        self.TDOM_Cam_savetimes = 0

        # 用于存储是否需要进行GS可见度重新设置
        self.ResetOpacityTime = 0

        # 用于标记是否需要进行分块训练
        self.Drone_Image_Block = False
        self.Progressive_Training_Block = False

        # 表示当前模型被分为多少个部分进行了训练，初始为1，只有在内存不够时会创建新的块
        self.BlockNum = 1

        # 表示每一个块对应的pth文件被存储在了哪个位置
        self.BlockDirPath = []

        self.resolution_scales = [1.0]

        # 设置初始场景的文件路径
        self.args.model_path = self.model_path_list[0] if self.args.ContinueFromImageNo == -1 else self.model_path_list[self.args.ContinueFromImageNo - self.args.StartFromImageNo]
        self.args.source_path = self.source_path_list[0] if self.args.ContinueFromImageNo == -1 else self.source_path_list[self.args.ContinueFromImageNo - self.args.StartFromImageNo]
        self.Diary.write(f"First scene source_path: {self.args.source_path}\nFirst scene model_path: {self.args.model_path}\n")

    # 为某一部分影像创建梯度图
    def GetGradLogForCams(self, cams=[], save=False):
        if cams == []:
            return 0

        for i in range(len(cams)):
            sys.stdout.write('\r')
            sys.stdout.write("Getting GradLog {}/{}".format(i + 1, len(cams)))
            sys.stdout.flush()

            if cams[i].GradLog == None:
                cams[i].GradLog, cams[i].MaxGradLog = Get_LogImage(cams[i].original_image, cams[i].image_name,
                                                                   os.path.join(self.args.Model_Path_Dir,
                                                                                "GradLogTrain"), save=save)

        sys.stdout.write('\n')

    def GetGradLogForImage(self, image, image_name, MG, save=False):
        return Get_LogImage(image, image_name, os.path.join(self.args.Model_Path_Dir, "GradLogRender"), MaxGradLog=MG, save=save)

    # 根据新的稀疏点云来扩张高斯点云
    def ExpandingGS_From_SparsePCD2(self, TrainCameras, get_ply=False):
        if os.path.exists(os.path.join(self.args.source_path_second, "sparse")):
            CurrentImagesNames = list(TrainCameras.keys())
            train_cam_infos, test_cam_infos, nerf_normalization, point3D_ids, xys, tri = sceneLoadTypeCallbacks[
                "Single_Image1"](
                self.args.source_path_second,
                self.args.images, self.args.eval,
                Do_Get_Tri_Mask=self.args.Use_Tri_Mask,
                Image_Name="",
                CurrentImagesNames=CurrentImagesNames,
                Diary=self.Diary,
                Silence=self.Silence)
        else:
            assert False, "Could not recognize scene type!"

        time1 = time.time()
        prepare_output_and_logger(self.args, True, Silence=self.Silence)

        # 初始化两个用于存储相机信息的数组：
        json_cams = []
        camlist = []

        # 如果测试影像或者训练影像存在，则将他们都加入到camlist中去
        if test_cam_infos:
            camlist.extend(test_cam_infos)
        if train_cam_infos:
            camlist.extend(train_cam_infos)

        # 将camlist中的影像信息转换为JSON的模式，并存储在json_cams中
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))

        # 将json_cams中的数据存在cameras.json中
        with open(os.path.join(self.args.model_path_second, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

        # 将包裹所有相机的球的半径赋值给self.cameras_extent（相机分布范围）
        if self.args.spatial_lr_scale == 0.0:
            # self.scene.cameras_extent = nerf_normalization["radius"]
            cameras_extent = nerf_normalization["radius"]
        else:
            # self.scene.cameras_extent = self.args.spatial_lr_scale
            cameras_extent = self.args.spatial_lr_scale

        time2 = time.time()
        # print(f"Before Add New Images: {time2 - time1}s")
        self.Diary.write(f"Before Add New Images: {time2 - time1}s\n")

        # 将新加入的影像读入到系统之中
        New_scene_info_traincam, NewCams, UpdatedTrainCamsPos = self.AddNewImages2(train_cam_infos, TrainCameras, threshold=self.args.LogMaskThreshold)

        scene_info = sceneLoadTypeCallbacks["Single_Image2"](self.args.source_path_second,
                                                             train_cam_infos, test_cam_infos, nerf_normalization,
                                                             point3D_ids, xys, tri, NewCams,
                                                             self.args.OriginImageHeight, self.args.OriginImageWidth,
                                                             get_ply=get_ply,
                                                             points_per_triangle=self.args.points_per_triangle,
                                                             device="cuda", Diary=self.Diary, Silence=self.Silence)

        time1 = time.time()
        if scene_info.ply_path is not None:
            with open(scene_info.ply_path, 'rb') as src_file, open(
                    os.path.join(self.args.model_path_second, "input.ply"), 'wb') as dest_file:
                dest_file.write(src_file.read())

        time2 = time.time()
        if not self.Silence:
            print(f"Copy Ply File: {time2 - time1}s")
        self.Diary.write(f"Copy Ply File: {time2 - time1}s\n")
        '''
        # 对于当前新影像，重置该影像内全部高斯球的可见度，然后进行一次剔除
        for newcam in NewCams:
            render_pkg = render(newcam, self.gaussians, self.args, self.bg, Render_Mask=newcam.Tri_Mask)'''

        # # 根据扩张后的稀疏点云来更新高斯点云
        # time1 = time.time()
        # self.gaussians.expand_from_pcd(scene_info.point_cloud, self.scene.cameras_extent)
        # time2 = time.time()
        # print(f"expand_from_pcd: {time2 - time1}s")
        # self.Diary.write(f"expand_from_pcd: {time2 - time1}s\n")
        #
        # # 记录这一次的稀疏点云
        # self.scene.basic_pcd = scene_info.point_cloud
        #
        # # 更新模型输出位置
        # self.scene.model_path = self.args.model_path_second

        return New_scene_info_traincam, NewCams, UpdatedTrainCamsPos, scene_info, self.args.model_path_second, cameras_extent

    # 读入新的影像数据并更新所有已有影像的位姿信息
    def AddNewImages2(self, train_cam_infos, TrainCameras, threshold=0.1):
        time1 = time.time()

        # 为self.train_cameras进行更新，将新加入的影像的相关信息加入到其中
        # 先找到新加入的影像
        NEW_TrainCamInfos = train_cam_infos
        CurrentNewImagesNum = 0
        NewImageNames = "["
        NewCamInfos = []
        for i in range(len(NEW_TrainCamInfos)):
            if NEW_TrainCamInfos[i].image is not None:
                New_CamInfo = NEW_TrainCamInfos[i]
                NewImageNames = NewImageNames + New_CamInfo.image_name + ", "
                NewCamInfos.append(New_CamInfo)
                CurrentNewImagesNum = CurrentNewImagesNum + 1
            if CurrentNewImagesNum == self.NewImagesNum:
                break

        NewImageNames = NewImageNames + "]"
        if not self.Silence:
            print(f"Newly Added Images Name: {NewImageNames}")
        self.Diary.write(f"Newly Added Images Name: {NewImageNames}\n")

        # 在每一个resolution_scale下：
        NewCams = []
        UpdatedTrainCamsPos = {}
        for resolution_scale in self.resolution_scales:
            # 更新self.train_cameras中[resolution_scale]每一个影像的位姿信息
            # for i in range(len(self.scene.train_cameras[resolution_scale])):
            for cam in NEW_TrainCamInfos:
                if (cam.image_name in TrainCameras):
                    world_view_transform = torch.tensor(getWorld2View2(cam.R, cam.T, TrainCameras[cam.image_name]["trans"], TrainCameras[cam.image_name]["scale"])).transpose(0, 1) # 4*4
                    projection_matrix = getProjectionMatrix(znear=TrainCameras[cam.image_name]["znear"],
                                                            zfar=TrainCameras[cam.image_name]["zfar"],
                                                            fovX=TrainCameras[cam.image_name]["FoVx"],
                                                            fovY=TrainCameras[cam.image_name]["FoVy"]).transpose(0, 1) # 4*4
                    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                    camera_center = world_view_transform.inverse()[3, :3]

                    UpdatedTrainCamsPos[cam.image_name] = [world_view_transform, projection_matrix, full_proj_transform, camera_center]

            # for i in range(len(TrainCameras)):
            #     for cam in NEW_TrainCamInfos:
            #         if (cam.image_name == TrainCameras[i].image_name):
            #             # TrainCameras[i].R = cam.R
            #             # TrainCameras[i].T = cam.T
            #         # if (cam.image_name == self.scene.train_cameras[resolution_scale][i].image_name):
            #         #     self.scene.train_cameras[resolution_scale][i].R = cam.R
            #         #     self.scene.train_cameras[resolution_scale][i].T = cam.T
            #
            #
            #             world_view_transform = torch.tensor(getWorld2View2(cam.R, cam.T, TrainCameras[i].trans, TrainCameras[i].scale)).transpose(0, 1)
            #             projection_matrix = getProjectionMatrix(znear=TrainCameras[i].znear, zfar=TrainCameras[i].zfar, fovX=TrainCameras[i].FoVx, fovY=TrainCameras[i].FoVy).transpose(0, 1)
            #             full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            #             camera_center = world_view_transform.inverse()[3, :3]
            #
            #             UpdatedTrainCamsPos[TrainCameras[i].image_name] = [world_view_transform, projection_matrix, full_proj_transform, camera_center]
            #
            #             break

            for cam_info in NewCamInfos:
                orig_w, orig_h = cam_info.image.size
                if self.args.resolution in [1, 2, 4, 8]:
                    resolution = round(orig_w / (resolution_scale * self.args.resolution)), round(
                        orig_h / (resolution_scale * self.args.resolution))
                else:  # should be a type that converts to float
                    if self.args.resolution == -1:
                        if orig_w > 1600:
                            global WARNED
                            if not WARNED and not self.Silence:
                                print(
                                    "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                                    "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                                WARNED = True
                            global_down = orig_w / 1600
                        else:
                            global_down = 1
                    else:
                        global_down = orig_w / self.args.resolution

                    scale = float(global_down) * float(resolution_scale)
                    resolution = (int(orig_w / scale), int(orig_h / scale))

                resized_image_rgb = PILtoTorch(cam_info.image, resolution)

                gt_image = resized_image_rgb[:3, ...]
                loaded_mask = None

                if resized_image_rgb.shape[1] == 4:
                    loaded_mask = resized_image_rgb[3:4, ...]

                # NewCam = Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                #                 FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                #                 image=gt_image, gt_alpha_mask=loaded_mask, Tri_Mask=cam_info.Tri_Mask,
                #                 image_name=cam_info.image_name, uid=len(self.scene.train_cameras[resolution_scale]),
                #                 data_device=self.args.data_device)
                NewCam = Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                                FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                                image=gt_image, gt_alpha_mask=loaded_mask, Tri_Mask=cam_info.Tri_Mask,
                                image_name=cam_info.image_name, uid=len(TrainCameras),
                                data_device=self.args.data_device)

                # self.scene.train_cameras[resolution_scale].append(NewCam)
                # TrainCameras.append(NewCam)

                self.GetGradLogForCams([NewCam])

                with torch.no_grad():
                    # render_pkg = render(NewCam, self.gaussians, self.args, self.bg, Render_Mask=NewCam.Tri_Mask)
                    # image = render_pkg["render"]

                    # 像后端发送指令，令其渲染一张当前新影像
                    msg = ["Render_New", NewCam]
                    self.GaussianTrainer_queue.put(msg)

                    while True:
                        if not self.DataPreProccesser_queue.empty():
                            image = self.DataPreProccesser_queue.get()[0]
                            break

                    renderGradLog, _ = self.GetGradLogForImage(image, NewCam.image_name, NewCam.MaxGradLog)

                    Log_minus = LogImage_minus(NewCam.GradLog, renderGradLog,
                                               os.path.join(self.args.Model_Path_Dir, "GradLogMinus"),
                                               NewCam.image_name)

                    NewCam.LogMask = Log_minus >= threshold

                NewCams.append(NewCam)

                if self.args.Block or self.ChosenImageNames != []:
                    self.ChosenImageNames.append(NewCam.image_name)

            # self.scene.scene_info_traincam = train_cam_infos
            New_scene_info_traincam = train_cam_infos

            time2 = time.time()
            # print(f"Add New Images: {time2 - time1}s")
            self.Diary.write(f"Add New Images: {time2 - time1}s\n")

            return New_scene_info_traincam, NewCams, UpdatedTrainCamsPos

    # 该函数将被送入到一个线程中，用于渐进式训练中的数据读取以及预处理
    def run(self):
        self.GaussianTrainer_queue.put(["init"])

        StartFrom = 0 if self.args.ContinueFromImageNo == -1 else self.args.ContinueFromImageNo - self.args.StartFromImageNo
        if StartFrom > len(self.source_path_list) - 2:
            return 0

        ProgressiveTrainingTime = StartFrom
        while True:
            # 在后端没有发送任何指令之前，不做任何行为
            if self.DataPreProccesser_queue.empty():
                continue
            else:
                Commond_from_GaussianTrainer = self.DataPreProccesser_queue.get()
                if Commond_from_GaussianTrainer[0] == "Progressive_First_Step":
                    if ProgressiveTrainingTime + self.args.StartFromImageNo + 1 <= self.args.EndAtImageNo or self.args.EndAtImageNo == -1:
                        # 数据更新同时计算这一次有多少新增影像
                        self.NewImagesNum = self.IM[ProgressiveTrainingTime + 1] - self.IM[ProgressiveTrainingTime]
                        self.args.source_path = self.source_path_list[ProgressiveTrainingTime]
                        self.args.model_path = self.model_path_list[ProgressiveTrainingTime]
                        self.args.source_path_second = self.source_path_list[ProgressiveTrainingTime + 1]
                        self.args.model_path_second = self.model_path_list[ProgressiveTrainingTime + 1]

                        # 根据新的稀疏点云来扩张高斯点云
                        get_ply = True if ProgressiveTrainingTime == len(self.source_path_list) - 3 else False
                        # self.NewCams = self.ExpandingGS_From_SparsePCD(get_ply=get_ply)
                        New_scene_info_traincam, NewCams, UpdatedTrainCamsPos, scene_info, new_model_path_second, cameras_extent = self.ExpandingGS_From_SparsePCD2(Commond_from_GaussianTrainer[1], get_ply=get_ply)
                        # print(f"ChosenImagesNum: {len(self.ChosenImageNames)}")

                        names = list(UpdatedTrainCamsPos.keys())
                        worlds = torch.stack([UpdatedTrainCamsPos[n][0] for n in names])  # (N,4,4)
                        projs = torch.stack([UpdatedTrainCamsPos[n][1] for n in names])  # (N,4,4)
                        fulls = torch.stack([UpdatedTrainCamsPos[n][2] for n in names])  # (N,4,4)
                        centers = torch.stack([UpdatedTrainCamsPos[n][3] for n in names])  # (N,3)

                        # 像后端发送指令和数据，开始对新进影像的训练
                        print("DataPreProccesser: Send Information to GaussianTrainer...")
                        # msg = ["Progressive_Train", New_scene_info_traincam, NewCams, TrainCameras, scene_info, new_model_path_second, cameras_extent, self.NewImagesNum]
                        msg = ["Progressive_Train", None, NewCams, [worlds, projs, fulls, centers, names], scene_info.point_cloud, new_model_path_second, cameras_extent, self.NewImagesNum]
                        self.GaussianTrainer_queue.put(msg)

                        ProgressiveTrainingTime += 1
                elif Commond_from_GaussianTrainer[0] == "Progressive":
                    if (ProgressiveTrainingTime + self.args.StartFromImageNo + 1 <= self.args.EndAtImageNo or self.args.EndAtImageNo == -1) and (ProgressiveTrainingTime < len(self.source_path_list) - 2):
                        # 数据更新同时计算这一次有多少新增影像
                        self.NewImagesNum = self.IM[ProgressiveTrainingTime + 1] - self.IM[ProgressiveTrainingTime]
                        self.args.source_path = self.source_path_list[ProgressiveTrainingTime]
                        self.args.model_path = self.model_path_list[ProgressiveTrainingTime]
                        self.args.source_path_second = self.source_path_list[ProgressiveTrainingTime + 1]
                        self.args.model_path_second = self.model_path_list[ProgressiveTrainingTime + 1]

                        # 根据新的稀疏点云来扩张高斯点云
                        get_ply = True if ProgressiveTrainingTime == len(self.source_path_list) - 3 else False
                        # self.NewCams = self.ExpandingGS_From_SparsePCD(get_ply=get_ply)
                        New_scene_info_traincam, NewCams, UpdatedTrainCamsPos, scene_info, new_model_path_second, cameras_extent = self.ExpandingGS_From_SparsePCD2(Commond_from_GaussianTrainer[1], get_ply=get_ply)
                        # print(f"ChosenImagesNum: {len(self.ChosenImageNames)}")

                        names = list(UpdatedTrainCamsPos.keys())
                        worlds = torch.stack([UpdatedTrainCamsPos[n][0] for n in names])  # (N,4,4)
                        projs = torch.stack([UpdatedTrainCamsPos[n][1] for n in names])  # (N,4,4)
                        fulls = torch.stack([UpdatedTrainCamsPos[n][2] for n in names])  # (N,4,4)
                        centers = torch.stack([UpdatedTrainCamsPos[n][3] for n in names])  # (N,3)

                        # 像后端发送指令和数据，开始对新进影像的训练，此时，需要注意先等后端把上一张影像完成训练之后，再进行传输
                        while True:
                            if not self.DataPreProccesser_queue.empty():
                                if self.DataPreProccesser_queue.get()[0] == "Progressive_finish":
                                    # 像后端发送指令和数据，开始对新进影像的训练
                                    print("DataPreProccesser: Send Information to GaussianTrainer...")
                                    msg = ["Progressive_Train", None, NewCams, [worlds, projs, fulls, centers, names], scene_info.point_cloud, new_model_path_second, cameras_extent, self.NewImagesNum]
                                    # msg = ["Progressive_Train"]
                                    self.GaussianTrainer_queue.put(msg)
                                break

                        ProgressiveTrainingTime += 1
                    else:
                        # 像后端发送指令和数据，开始进行最终的全局优化，此时，需要注意先等后端把上一张影像完成训练之后，再进行传输
                        while True:
                            if not self.DataPreProccesser_queue.empty():
                                if self.DataPreProccesser_queue.get()[0] == "Progressive_finish":
                                    print("DataPreProccesser: Start Final Refinement...")
                                    self.args.source_path = self.source_path_list[-2]
                                    self.args.model_path = self.model_path_list[-2]
                                    self.args.source_path_second = self.source_path_list[-1]
                                    self.args.model_path_second = self.model_path_list[-1]
                                    # self.scene.model_path = self.args.model_path_second
                                    prepare_output_and_logger(self.args, True)
                                    shutil.copy(os.path.join(self.args.model_path, "input.ply"), os.path.join(self.args.model_path_second, "input.ply"))
                                    shutil.copy(os.path.join(self.args.model_path, "cameras.json"), os.path.join(self.args.model_path_second, "cameras.json"))

                                    msg = ["Final_Refinement", self.args.model_path_second]
                                    self.GaussianTrainer_queue.put(msg)

                                    break
                elif Commond_from_GaussianTrainer[0] == "stop":
                    break

# 后端，第二个线程，主要用于高斯训练
class GaussianTrainer(mp.Process):
    # 构造函数
    def __init__(self):
        super().__init__()

        # 获取参数配置器
        self.args, self.lp, self.op, self.pp = GetArgs()

        self.ImagesAlreadyBeTrainedIterations = None
        self.ResetOpacityTime = 0

        # 用于两个进程之间的信息传递
        self.DataPreProccesser_queue = None
        self.GaussianTrainer_queue = None

        # 记录模型的已训练次数
        self.AlreadyTrainingIterations = 0
        self.StartFromTrainingIterations = 0

        self.Drone_Image_Block = False
        self.Progressive_Training_Block = False

        self.source_path_list, self.model_path_list = None, None
        self.resolution_scales = [1.0]

        # 该字典用于存储每一张影像对应的高斯球
        self.Image_Visibility = {}
        self.MaxWeightsImages = []

        # 该列表用于存储一些影像的名称，仅在需要进行分块训练时会被用到
        # 若该列表不为空，则训练影像时只会从该列表中的影像来选取
        self.ChosenImageNames = []

        self.NewCams = []

        self.commond = None

        self.imagenum = self.args.StartFromImageNo

    def InitialGaussianTrain(self):
        # 设置初始场景的文件路径
        self.args.model_path = self.model_path_list[0] if self.args.ContinueFromImageNo == -1 else self.model_path_list[self.args.ContinueFromImageNo - self.args.StartFromImageNo]
        self.args.source_path = self.source_path_list[0] if self.args.ContinueFromImageNo == -1 else self.source_path_list[self.args.ContinueFromImageNo - self.args.StartFromImageNo]
        # self.Diary.write(f"First scene source_path: {self.args.source_path}\nFirst scene model_path: {self.args.model_path}\n")

        # 初始化背景颜色
        self.bg_color = [1, 1, 1] if self.args.white_background else [0, 0, 0]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")
        self.bg = torch.rand((3), device="cuda") if self.args.random_background else self.background

        # 初始化log损失累计
        self.ema_loss_for_log = 0.0

        # 主要进行一系列的模型以及其它文件的输出准备（例如构建输出文件夹等），并返回一个Tensorboard writer对象
        self.tb_writer = prepare_output_and_logger(self.args)

        # 初始化3D Gaussian Splatting模型，主要是一些模型属性的初始化和神经网络中一些激活函数的初始化
        self.gaussians = GaussianModel(self.args.sh_degree)

        # 初始化一个场景
        self.scene = Scene(self.args, self.gaussians, load_iteration=None, shuffle=True, resolution_scales=[1.0], Do_Get_Tri_Mask=self.args.Use_Tri_Mask)

        # 为当前已经读入的影像全部创建一个梯度图
        self.GetGradLogForCams()

        # 进行一些训练上的初始化设置
        self.gaussians.training_setup(self.args)

        # 进行初始模型的训练
        if (not self.args.skip_FirstSceneTraining) and self.args.ContinueFromImageNo == -1:
            self.Train_Gaussians(self.args.IterationFirstScene, "Initialization")

        # 向前端传递信息，表示初始训练已经完成，可以开始读取后续的影像
        self.SendImagePos_to_DataPreProccesser(True)

    # 为某一部分影像创建梯度图
    def GetGradLogForCams(self, cams=[], save=False):
        if cams == []:
            cams = self.scene.getTrainCameras()

        for i in range(len(cams)):
            sys.stdout.write('\r')
            sys.stdout.write("Getting GradLog {}/{}".format(i + 1, len(cams)))
            sys.stdout.flush()

            if cams[i].GradLog == None:
                cams[i].GradLog, cams[i].MaxGradLog = Get_LogImage(cams[i].original_image, cams[i].image_name, os.path.join(self.args.Model_Path_Dir, "GradLogTrain"), save=save)

        sys.stdout.write('\n')

    def ImagesAlreadyBeTrainedIterations_Set(self):
        for NewCam in self.NewCams:
            Equivalent_training_times = 0
            self.ImagesAlreadyBeTrainedIterations[NewCam.image_name] = Equivalent_training_times
            # print(f"{NewCam.image_name}: Equavelent Training Times = {Equivalent_training_times}")

    # 将现有影像的位姿信息传给前端
    def SendImagePos_to_DataPreProccesser(self, IsFirstStep=False):
        TrainCamsDict = {}
        for cam in self.scene.getTrainCameras():
            TrainCamsDict[cam.image_name] = {"trans": cam.trans,
                                             "scale": cam.scale,
                                             "znear": cam.znear,
                                             "zfar": cam.zfar,
                                             "FoVx": cam.FoVx,
                                             "FoVy": cam.FoVy}

        # 向前端传递信息，表示开始读取后续的影像
        if IsFirstStep:
            msg = ["Progressive_First_Step", TrainCamsDict]
            self.DataPreProccesser_queue.put(msg)
        else:
            msg = ["Progressive", TrainCamsDict]
            self.DataPreProccesser_queue.put(msg)

    # 3DGS模型训练核心函数
    def Train_Gaussians(self, iteration, Training_Type):
        # self.Diary.write('\n')

        Start_From_Its = self.AlreadyTrainingIterations

        # 初始化影像栈以及进度条
        viewpoint_stack = None
        EvaluateRender = False
        progress_bar = tqdm(range(0, Start_From_Its + iteration), desc=f"{Training_Type} Training progress",
                            initial=Start_From_Its)

        sub_iteration = 1
        while sub_iteration <= iteration:
            # 如果需要给前端渲染影像，那么必须等这一张影像的训练至少进行一半才可以进行
            if not self.GaussianTrainer_queue.empty() or self.commond is not None:
                if self.commond is None:
                    self.commond = self.GaussianTrainer_queue.get()
                if self.commond[0] == "Render_New" and sub_iteration > iteration / 2:
                    with torch.no_grad():
                        NewCam = self.commond[1]
                        render_pkg = render(NewCam, self.gaussians, self.args, self.bg, Render_Mask=NewCam.Tri_Mask)
                        image = render_pkg["render"]

                        # 向前端传递信息
                        msg = [image]
                        self.DataPreProccesser_queue.put(msg)

                        self.commond = None
                elif self.commond[0] == "Render_New" and sub_iteration <= iteration / 2:
                    pass
                elif self.commond[0] == "Final_Refinement":
                    pass
                else:
                    print(self.commond)
                    self.commond = None


            # for sub_iteration in range(iteration):
            # 更新模型已训练次数
            self.AlreadyTrainingIterations += 1

            # 当摄影机视点栈堆为空时：
            if not viewpoint_stack:
                if Training_Type != "On_The_Fly":
                    # 将Scene类中所有的self.train_cameras中指定缩放比例的训练影像及其相关信息存入到摄影机视点栈堆中
                    viewpoint_stack = self.scene.getTrainCameras().copy()

                    # 如果是模型初始化训练，那么需要先将初始模型就已存在的影像的已训练次数置为0
                    if Training_Type == "Initialization" and len(
                            list(self.ImagesAlreadyBeTrainedIterations.keys())) == 0:
                        for _ in range(len(viewpoint_stack)):
                            self.ImagesAlreadyBeTrainedIterations[viewpoint_stack[_].image_name] = int(
                                self.args.Mean * self.args.IterationFirstScene / 30000)
                else:
                    # 在渐进式训练过程中，将根据特殊规则选取用于训练的影像
                    viewpoint_stack = self.GetTraining_Viewpoints_SingleImage(single=self.args.Single)

            # 随机从摄影机视点栈堆中任意取出一个影像以及其视点信息
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

            # 如果当前影像异常，应当跳过并返回一次训练次数
            if viewpoint_cam.Tri_Mask.shape[0] == 2:
                self.AlreadyTrainingIterations -= 1
                continue

            # 学习率下降
            if Training_Type == "Initialization":
                lr = self.gaussians.update_learning_rate(self.AlreadyTrainingIterations)
            else:
                lr = self.gaussians.SingleImage_Update_Lr(self.ImagesAlreadyBeTrainedIterations,
                                                          self.args, viewpoint_cam,
                                                          self.args.IterationPerMergeScene + self.args.GlobalOptimizationIteration)

            # 将球谐函数的阶数提高一阶，最多至3阶
            if self.AlreadyTrainingIterations % 1000 == 0:
                self.gaussians.oneupSHdegree()

            # 在进行渐进式训练前，保证球谐函数已经提升至最高
            if self.gaussians.active_sh_degree < self.args.sh_degree:
                self.gaussians.oneupSHdegree()

            # 计算损失
            if Training_Type != "On_The_Fly":
                # 进行影像渲染，渲染指定视点的影像
                render_pkg = render(viewpoint_cam, self.gaussians, self.args, self.bg,
                                    Render_Mask=viewpoint_cam.Tri_Mask)
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    load_distribution
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter_bool"],
                    render_pkg["radii"],
                    render_pkg["load_distribution"]
                )

                self.Image_Visibility[viewpoint_cam.image_name] = visibility_filter

                # 计算损失
                if self.args.Use_Tri_Mask:
                    image = image * viewpoint_cam.Tri_Mask
                    load_distribution = load_distribution[viewpoint_cam.Tri_Mask.squeeze(0)]

                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
                loss = (1.0 - self.args.lambda_dssim) * Ll1 + self.args.lambda_dssim * (1.0 - ssim_value)

                load_loss = torch.std(load_distribution)  # 计算标准差
                loss_adj = math.floor(math.log10(load_loss)) - math.floor(math.log10(loss))
                load_loss = load_loss / math.pow(10, loss_adj + 1.0)

                loss = loss * (1 - self.args.lambda_load) + self.args.lambda_load * load_loss
            # 目前暂时采用与其他情况相同的训练策略
            else:
                # 进行影像渲染，渲染指定视点的影像
                render_pkg = render(viewpoint_cam, self.gaussians, self.args, self.bg,
                                    Render_Mask=viewpoint_cam.Tri_Mask)
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    load_distribution
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter_bool"],
                    render_pkg["radii"],
                    render_pkg["load_distribution"]
                )

                self.Image_Visibility[viewpoint_cam.image_name] = visibility_filter

                # 计算损失
                if self.args.Use_Tri_Mask:
                    image = image * viewpoint_cam.Tri_Mask
                    load_distribution = load_distribution[viewpoint_cam.Tri_Mask.squeeze(0)]

                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
                loss = (1.0 - self.args.lambda_dssim) * Ll1 + self.args.lambda_dssim * (1.0 - ssim_value)

                load_loss = torch.std(load_distribution)  # 计算标准差
                loss_adj = math.floor(math.log10(load_loss)) - math.floor(math.log10(loss))
                load_loss = load_loss / math.pow(10, loss_adj + 1.0)

                loss = loss * (1 - self.args.lambda_load) + self.args.lambda_load * load_loss

            # 完成这一次训练后，让当前训练影像的已训练次数+1
            if Training_Type != "Initialization":
                self.ImagesAlreadyBeTrainedIterations[viewpoint_cam.image_name] = self.ImagesAlreadyBeTrainedIterations[
                                                                                      viewpoint_cam.image_name] + 1

            # self.GaussianDiary.write(f"Iteration: {self.AlreadyTrainingIterations}, "
            #                          f"Image Name: {viewpoint_cam.image_name}, "
            #                          f"Image Training Times: {self.ImagesAlreadyBeTrainedIterations[viewpoint_cam.image_name]}, "
            #                          f"Lr: {lr}, "
            #                          f"spatial_lr_scale: {self.scene.cameras_extent}, "
            #                          f"Gaussians: {visibility_filter.shape[0]}\n")

            # 反向传播
            loss.backward()

            with torch.no_grad():
                # 计算log损失累计
                self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log

                # 每十次训练更新一次进度条
                progress_bar.update(1)
                progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
                if sub_iteration == iteration:
                    EvaluateRender = True
                    progress_bar.close()

                GaussianPruned = False

                # 判断是否需要重新设置可见度（注意，此时在渐进式训练过程中暂时不进行该步骤）
                if self.AlreadyTrainingIterations % self.args.opacity_reset_interval == 0:
                    '''
                    if (Training_Type == "Initialization") or (
                            Training_Type == "Final_Refinement" and
                            self.args.FinalOptimizationIterations - (
                                    self.AlreadyTrainingIterations - Start_From_Its) > self.args.opacity_reset_interval and
                            self.AlreadyTrainingIterations <= Start_From_Its + iteration / 2):'''
                    if (Training_Type == "Initialization"):
                        self.ResetOpacityTime += 1

                # 如果处于模型初始化训练阶段
                if Training_Type == "Initialization" and self.AlreadyTrainingIterations < self.args.densify_until_iter:
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    # 稠密化
                    if self.AlreadyTrainingIterations > self.args.densify_from_iter and self.AlreadyTrainingIterations % self.args.densification_interval == 0 and self.ResetOpacityTime < 1:
                        size_threshold = 20 if self.AlreadyTrainingIterations > self.args.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(self.args.densify_grad_threshold, self.args.opacity_threshold,
                                                         self.scene.cameras_extent, size_threshold, radii)
                        GaussianPruned = True
                '''
                elif Training_Type == "Final_Refinement" and self.AlreadyTrainingIterations <= Start_From_Its + iteration / 2:
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    # 稠密化
                    if self.AlreadyTrainingIterations % self.args.densification_interval == 0 and self.ResetOpacityTime < 1:
                        size_threshold = 20 if self.AlreadyTrainingIterations > self.args.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(self.args.densify_grad_threshold, self.args.opacity_threshold, self.scene.cameras_extent, size_threshold, radii)
                        GaussianPruned = True'''

                # 重新设置可见度
                if self.ResetOpacityTime >= 1 and (not (self.Drone_Image_Block or self.Progressive_Training_Block)) and self.AlreadyTrainingIterations - self.StartFromTrainingIterations >= 100:
                    print(f"[{self.AlreadyTrainingIterations} its] => Reset Opacity!!")
                    self.gaussians.reset_opacity(self.Image_Visibility)

                    self.Image_Visibility = {}

                    self.ResetOpacityTime -= 1

                if GaussianPruned:
                    self.Image_Visibility = {}

                # 优化器参数更新
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

                # if sub_iteration + 1 == iteration:
                #     if Training_Type == "On_The_Fly":
                #         self.Phase_End("GS_Expanding")
                #     elif Training_Type == "Final_Refinement":
                #         self.Phase_End("Final_Refinement")
                #     elif Training_Type == "Initialization":
                #         self.Phase_End("Initial_GS_Optimization")

                # 每训练一定次数输出一次模型评估结果
                self.Evaluate(Training_Type, EvaluateRender)
                EvaluateRender = False

            sub_iteration = sub_iteration + 1

        # 在完成这一张影像的训练之后
        if Training_Type == "On_The_Fly":
            with torch.no_grad():
                # 进行一次高斯球剔除
                prune_mask = (self.gaussians.get_opacity < self.args.opacity_threshold).squeeze()
                self.gaussians.prune_points(prune_mask)
                print(f"Prune Gaussians Num: {prune_mask.sum()}")

    # 对当前高斯模型进行一次评估
    def Evaluate(self, Training_Type, EvaluateRender):
        DoEvaluate = False
        if Training_Type == "Initialization" and (
                self.AlreadyTrainingIterations % self.args.InitialTrainingEvaluateInterval == 0 or EvaluateRender):
            DoEvaluate = True
        elif (Training_Type == "On_The_Fly" or Training_Type == "Final_Refinement") and EvaluateRender:
            DoEvaluate = True

            if Training_Type == "Final_Refinement":
                self.ChosenImageNames = []
        elif Training_Type == "Final_Refinement" and self.AlreadyTrainingIterations % self.args.FinalTrainingEvaluateInterval == 0:
            DoEvaluate = True

        if self.args.NoDebug and Training_Type != "Final_Refinement":
            DoEvaluate = False

        if DoEvaluate:
            self.EvaluateModel(l1_loss, render, (self.pp.extract(self.args), self.background))

    # 打印模型的评价结果
    def EvaluateModel(self, l1_loss, renderFunc, renderArgs):
        # Report test and samples of training set
        torch.cuda.empty_cache()

        if self.ChosenImageNames == []:
            validation_configs = ({'name': 'test', 'cameras': self.scene.getTestCameras()},
                                  {'name': 'train', 'cameras': self.scene.getTrainCameras()})
        else:
            validation_configs = ({'name': 'test', 'cameras': self.scene.getTestCameras()},
                                  {'name': 'train', 'cameras': [traincam for traincam in self.scene.getTrainCameras() if
                                                                traincam.image_name in self.ChosenImageNames]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0

                PSNRs = []
                ErrorImageNum = 0
                for idx, viewpoint in enumerate(config['cameras']):
                    if viewpoint.Tri_Mask.shape[0] != 2:
                        image = torch.clamp(
                            renderFunc(viewpoint, self.gaussians, *renderArgs, Render_Mask=viewpoint.Tri_Mask)[
                                "render"], 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        l1_test += l1_loss(image, gt_image).mean().double()

                        PSNR = psnr(image, gt_image, viewpoint.Tri_Mask).mean().double()
                        psnr_test += PSNR
                        PSNRs.append(PSNR)
                    else:
                        ErrorImageNum = ErrorImageNum + 1

                psnr_test /= len(config['cameras']) - ErrorImageNum
                l1_test /= len(config['cameras']) - ErrorImageNum
                print("\n[ITER {}] Evaluating {}: L1 {} Gaussians {} PSNR {}".format(self.AlreadyTrainingIterations,
                                                                                     config['name'], l1_test,
                                                                                     self.gaussians.get_xyz.shape[0],
                                                                                     psnr_test))
                # self.Diary.write(
                #     "[ITER {}] Evaluating {}: L1 {} Gaussians {} PSNR {}\n".format(self.AlreadyTrainingIterations,
                #                                                                    config['name'], l1_test,
                #                                                                    self.gaussians.get_xyz.shape[0],
                #                                                                    psnr_test))
                #
                # if self.EvaluateDiary is not None and len(PSNRs) != 0:
                #     self.EvaluateDiary.write("-----------------------------------------------------------\n")
                #     self.EvaluateDiary.write(f"Iterations: {self.AlreadyTrainingIterations}\n")
                #     for i in range(len(PSNRs)):
                #         self.EvaluateDiary.write(f"{config['cameras'][i].image_name}: {PSNRs[i]}\n")

    # 仅训练当前新传入的影像
    def GetTraining_Viewpoints_SingleImage(self, single=False):
        viewpoint_stack = []
        if single:
            for i in range(self.args.IterationPerMergeScene + self.args.GlobalOptimizationIteration):
                viewpoint_stack.append(self.NewCams[0])
        else:
            for i in range(self.args.IterationPerMergeScene):
                viewpoint_stack.append(self.NewCams[0])

            ALL_viewpoint_stack = self.scene.getTrainCameras().copy()
            ImageIndex = [j for j in range(len(ALL_viewpoint_stack))]
            RestImageNeed = self.args.GlobalOptimizationIteration
            while RestImageNeed > 0:
                if len(ImageIndex) > RestImageNeed:
                    SampleIndex = random.sample(ImageIndex, RestImageNeed)
                    for i in SampleIndex:
                        viewpoint_stack.append(ALL_viewpoint_stack[i])
                    RestImageNeed = 0
                else:
                    for i in ImageIndex:
                        viewpoint_stack.append(ALL_viewpoint_stack[i])
                        RestImageNeed = RestImageNeed - 1
                        if RestImageNeed == 0:
                            break

        return viewpoint_stack

    # 为所有影像进行一次渲染并输出一些评估结果
    def Render_Evaluate_All_Images(self):
        with torch.no_grad():
            ALL_viewpoint_stack = self.scene.getTrainCameras().copy()
            All_Predicted_Images = []
            ALL_Image_Visibility_Filter = []
            ALL_Image_PSNR = []
            progress_bar = tqdm(range(len(ALL_viewpoint_stack)), desc="Rendering progress")
            for i in range(len(ALL_viewpoint_stack)):
                ImageOutputDir = os.path.join(self.args.Model_Path_Dir, "OutputImages", ALL_viewpoint_stack[i].image_name)
                PSNR_File_Path = os.path.join(ImageOutputDir, "PSNR.txt")
                os.makedirs(ImageOutputDir, exist_ok=True)
                PSNR_File = open(PSNR_File_Path, 'w')

                progress_bar.update(1)

                viewpoint_cam = ALL_viewpoint_stack[i]
                render_pkg = render(viewpoint_cam, self.gaussians, self.args, self.bg,
                                    Render_Mask=torch.ones(1, dtype=torch.bool).cuda())
                image, visibility_filter = render_pkg["render"], render_pkg["visibility_filter"]

                All_Predicted_Images.append(image)
                ALL_Image_Visibility_Filter.append(visibility_filter)

                ImageOutput = (image - image.min()) / (image.max() - image.min())
                ImageOutput = ImageOutput.permute(1, 2, 0).cpu().numpy()
                ImageOutput = (ImageOutput * 255).astype(np.uint8)
                ImageOutput = Image.fromarray(ImageOutput)
                ImageOutput.save(os.path.join(ImageOutputDir, f"PredictionImages{self.AlreadyTrainingIterations}.jpg"))

                gt_image = viewpoint_cam.original_image.cuda()
                PSNR = psnr(image, gt_image)
                PSNR = PSNR.cpu().sum().item() / 3
                ALL_Image_PSNR.append(PSNR)

                PSNR_File.write(str(self.AlreadyTrainingIterations) + f": {PSNR}, Visible_Gaussians_Num: {visibility_filter.shape[0]}\n")
                PSNR_File.close()

    # 该函数将被送入到一个线程中，用于渐进式训练中的数据读取以及预处理
    def run(self):
        while True:
            if self.GaussianTrainer_queue.empty():
                continue
            else:
                Commond_from_DataPreProccesser = self.GaussianTrainer_queue.get()
                if Commond_from_DataPreProccesser[0] == "init":
                    self.InitialGaussianTrain()
                elif Commond_from_DataPreProccesser[0] == "stop":
                    break
                elif Commond_from_DataPreProccesser[0] == "Render_New":
                    with torch.no_grad():
                        NewCam = Commond_from_DataPreProccesser[1]
                        render_pkg = render(NewCam, self.gaussians, self.args, self.bg, Render_Mask=NewCam.Tri_Mask)
                        image = render_pkg["render"]

                        # 向前端传递信息
                        msg = [image]
                        self.DataPreProccesser_queue.put(msg)
                elif Commond_from_DataPreProccesser[0] == "Progressive_Train":
                    self.imagenum += 1
                    print(f"-----------------------------This is {self.imagenum} image-----------------------------")
                    print("GaussianTrainer: Get Information from DataPreProccesser...")

                    self.scene.scene_info_traincam = Commond_from_DataPreProccesser[1]
                    self.NewCams = Commond_from_DataPreProccesser[2]
                    UpdatedTrainCamsPosList = Commond_from_DataPreProccesser[3]
                    self.scene.cameras_extent = Commond_from_DataPreProccesser[6]
                    self.gaussians.expand_from_pcd(Commond_from_DataPreProccesser[4], self.scene.cameras_extent)
                    self.scene.basic_pcd = Commond_from_DataPreProccesser[4]
                    self.scene.model_path = Commond_from_DataPreProccesser[5]
                    NewImagesNum = Commond_from_DataPreProccesser[7]

                    # 还原为 dict
                    worlds = UpdatedTrainCamsPosList[0]
                    projs = UpdatedTrainCamsPosList[1]
                    fulls = UpdatedTrainCamsPosList[2]
                    centers = UpdatedTrainCamsPosList[3]
                    names = UpdatedTrainCamsPosList[4]
                    UpdatedTrainCamsPos = {}
                    for i, name in enumerate(names):
                        UpdatedTrainCamsPos[name] = [
                            worlds[i],
                            projs[i],
                            fulls[i],
                            centers[i]
                        ]

                    # 更新位姿
                    for i in range(len(self.scene.train_cameras[1.0])):
                        image_name = self.scene.train_cameras[1.0][i].image_name
                        self.scene.train_cameras[1.0][i].world_view_transform = UpdatedTrainCamsPos[image_name][0].cuda()
                        self.scene.train_cameras[1.0][i].projection_matrix = UpdatedTrainCamsPos[image_name][1].cuda()
                        self.scene.train_cameras[1.0][i].full_proj_transform = UpdatedTrainCamsPos[image_name][2].cuda()
                        self.scene.train_cameras[1.0][i].camera_center = UpdatedTrainCamsPos[image_name][3].cuda()
                    self.scene.train_cameras[1.0].append(Commond_from_DataPreProccesser[2][0])

                    self.ImagesAlreadyBeTrainedIterations_Set()

                    # 向前端传递信息，表示开始读取后续的影像
                    self.SendImagePos_to_DataPreProccesser()

                    # 对扩张后的场景进行训练
                    if not self.args.skip_MergeSceneTraining:
                        self.Train_Gaussians((self.args.IterationPerMergeScene + self.args.GlobalOptimizationIteration) * NewImagesNum, "On_The_Fly")

                    # 向前端传递信息，表示这一张影像已经完成训练，可以将下一张影像的相关信息传输到后端
                    msg = ["Progressive_finish"]
                    self.DataPreProccesser_queue.put(msg)
                elif Commond_from_DataPreProccesser[0] == "Final_Refinement":
                    self.scene.model_path = Commond_from_DataPreProccesser[1]
                    if not self.args.skip_GlobalOptimization:
                        self.Train_Gaussians(self.args.FinalOptimizationIterations, "Final_Refinement")

                    self.Render_Evaluate_All_Images()

                    # 向前端传递信息
                    msg = ["stop"]
                    self.DataPreProccesser_queue.put(msg)

# 运行主要函数
class On_the_Fly_GS:
    def __init__(self):
        self.DataPreProccesser_queue = mp.Queue()
        self.GaussianTrainer_queue = mp.Queue()

        self.DataPreProccesser = DataPreProccesser()
        self.DataPreProccesser.DataPreProccesser_queue = self.DataPreProccesser_queue
        self.DataPreProccesser.GaussianTrainer_queue = self.GaussianTrainer_queue

        self.GaussianTrainer = GaussianTrainer()
        self.GaussianTrainer.DataPreProccesser_queue = self.DataPreProccesser_queue
        self.GaussianTrainer.GaussianTrainer_queue = self.GaussianTrainer_queue
        self.GaussianTrainer.ImagesAlreadyBeTrainedIterations = self.DataPreProccesser.ImagesAlreadyBeTrainedIterations
        self.GaussianTrainer.source_path_list = self.DataPreProccesser.source_path_list
        self.GaussianTrainer.model_path_list = self.DataPreProccesser.model_path_list

        print("Start Thread...\n")
        self.GaussianTrainer_Thread = mp.Process(target=self.GaussianTrainer.run)
        self.GaussianTrainer_Thread.start()
        self.DataPreProccesser.run()

        self.GaussianTrainer_queue.put(["stop"])
        self.GaussianTrainer_Thread.join()
        print("\nStart Thread...")

if __name__ == "__main__":
    mp.set_start_method("spawn")

    Run = On_the_Fly_GS()