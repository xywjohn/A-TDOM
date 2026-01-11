#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text, read_points3D_ID_binary
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import time
from scipy.spatial import Delaunay
import cv2

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    Tri_Mask: torch.Tensor

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def compute_barycentric_coords_torch(p, a, b, c):
    """
    输入：
        p: (N, 2)
        a, b, c: (N, 2)
    输出：
        bary_coords: (N, 3)
    """
    v0 = b - a  # (N, 2)
    v1 = c - a
    v2 = p - a

    d00 = torch.sum(v0 * v0, dim=1)  # (N,)
    d01 = torch.sum(v0 * v1, dim=1)
    d11 = torch.sum(v1 * v1, dim=1)
    d20 = torch.sum(v2 * v0, dim=1)
    d21 = torch.sum(v2 * v1, dim=1)

    denom = d00 * d11 - d01 * d01  # (N,)

    # 避免除以 0（退化三角形）
    eps = 1e-8
    denom = torch.where(denom == 0, torch.ones_like(denom) * eps, denom)

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return torch.stack([u, v, w], dim=1)  # (N, 3)

def Get_3D_Sampled_Points_GPU(sampled_points, Tri_IDs, tri_simplices, xys, corresponding_3d_points, corresponding_3d_points_rgb, sampled_corresponding_3d_points, sampled_corresponding_3d_points_rgb):
    """
        使用 PyTorch 实现 GPU 加速的 3D 插值。

        输入：
            sampled_points: (M, 2) float tensor
            Tri_IDs: (M,) long tensor，表示每个采样点在哪个三角形里
            tri_simplices: (T, 3) long tensor，三角形索引
            xys: (N, 2) float tensor，2D点
            corresponding_3d_points: (N, 3) float tensor
            corresponding_3d_points_rgb: (N, 3) float tensor

        输出：
            all_points_3d: (N+M, 3)
            all_colors: (N+M, 3)

        """
    if Tri_IDs is not None:
        device = sampled_points.device

        triangle_indices = tri_simplices[Tri_IDs]  # (M, 3)

        # 获取三角形三个顶点的 2D 坐标 (M, 3, 2)
        triangle_xys = xys[triangle_indices]  # (M, 3, 2)

        # triangle_indices_np = triangle_indices.cpu().numpy()  # 转成 CPU numpy 索引
        # triangle_xys_np = xys[triangle_indices_np]  # 用 numpy 进行索引 (M, 3, 2)
        # triangle_xys = torch.from_numpy(triangle_xys_np).to(device=device, dtype=torch.float32)  # 转为 GPU tensor

        a2d, b2d, c2d = triangle_xys[:, 0], triangle_xys[:, 1], triangle_xys[:, 2]

        # 获取对应的 3D 坐标 (M, 3, 3)
        triangle_3d = corresponding_3d_points[triangle_indices]  # (M, 3, 3)
        a3d, b3d, c3d = triangle_3d[:, 0], triangle_3d[:, 1], triangle_3d[:, 2]

        # 获取对应的颜色 (M, 3, 3)
        triangle_rgb = corresponding_3d_points_rgb[triangle_indices]
        a_rgb, b_rgb, c_rgb = triangle_rgb[:, 0], triangle_rgb[:, 1], triangle_rgb[:, 2]

        # 计算重心坐标
        bary_coords = compute_barycentric_coords_torch(sampled_points, a2d, b2d, c2d)  # (M, 3)
        bary_coords = bary_coords.to(dtype=torch.float32, device=device)

        # 插值
        interpolated_3d = (
                bary_coords[:, 0:1] * a3d +
                bary_coords[:, 1:2] * b3d +
                bary_coords[:, 2:3] * c3d
        )

        interpolated_rgb = (
                bary_coords[:, 0:1] * a_rgb +
                bary_coords[:, 1:2] * b_rgb +
                bary_coords[:, 2:3] * c_rgb
        )

        if sampled_corresponding_3d_points is not None:
            # 合并
            all_points_3d = torch.cat([sampled_corresponding_3d_points, interpolated_3d], dim=0)
            all_colors = torch.cat([sampled_corresponding_3d_points_rgb, interpolated_rgb], dim=0)
        else:
            all_points_3d = interpolated_3d
            all_colors = interpolated_rgb

    elif sampled_corresponding_3d_points is not None:
        all_points_3d = sampled_corresponding_3d_points
        all_colors = sampled_corresponding_3d_points_rgb

    else:
        all_points_3d = None
        all_colors = None

    return all_points_3d, all_colors

def compute_barycentric_coords(p, a, b, c):
    """
    计算点 p 相对于三角形 abc 的重心坐标
    p, a, b, c 都是 2D 点 (x, y)
    返回 (w1, w2, w3) 重心权重
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    if denom == 0:
        # 退化三角形，返回默认值
        return np.array([1.0, 0.0, 0.0])

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return np.array([u, v, w])

def Get_3D_Sampled_Points(sampled_points, Tri_IDs, tri, xys, corresponding_3d_points, corresponding_3d_points_rgb, sampled_corresponding_3d_points, sampled_corresponding_3d_points_rgb):
    time1 = time.time()

    # 输出采样点对应的3D坐标
    interpolated_3d_points = []
    interpolated_colors = []

    for i, sample_pt in enumerate(sampled_points):
        tri_idx = Tri_IDs[i]
        triangle = tri.simplices[tri_idx]  # 三角形顶点索引 (3,)

        # 从2D点集中获取三角形三个顶点的2D坐标
        a2d, b2d, c2d = xys[triangle]
        # 对应的3D坐标
        a3d, b3d, c3d = corresponding_3d_points[triangle]
        # RGB颜色，形状应为 (3,)
        a_color, b_color, c_color = corresponding_3d_points_rgb[triangle]

        # 计算该采样点在三角形中的重心坐标
        bary_coords = compute_barycentric_coords(sample_pt, a2d, b2d, c2d)

        # 用重心坐标对3D点插值
        interpolated_point = (
                bary_coords[0] * a3d +
                bary_coords[1] * b3d +
                bary_coords[2] * c3d
        )

        # 对颜色进行插值（R/G/B 同时插）
        interpolated_color = (
                bary_coords[0] * a_color +
                bary_coords[1] * b_color +
                bary_coords[2] * c_color
        )

        interpolated_3d_points.append(interpolated_point)
        interpolated_colors.append(interpolated_color)

    interpolated_3d_points = np.array(interpolated_3d_points)  # shape (M, 3)
    interpolated_colors = np.array(interpolated_colors)

    time2 = time.time()

    print(f"3D Sample Points: {time2 - time1}s, Sampled Point Number: {interpolated_3d_points.shape[0]}")

    # 合并 3D 点
    all_points_3d = np.vstack([sampled_corresponding_3d_points, interpolated_3d_points])  # shape (N+M, 3)

    # 合并颜色
    all_colors = np.vstack([sampled_corresponding_3d_points_rgb, interpolated_colors])  # shape (N+M, 3)

    return all_points_3d, all_colors

def sample_points_in_triangle_torch(v0, v1, v2, n):
    """
    使用 PyTorch 在三角形内均匀采样 n 个点（支持 GPU）。

    Args:
        v0, v1, v2: torch.Tensor of shape (2,)，三角形顶点坐标，必须在同一 device（CPU 或 GPU）
        n: int，采样点数

    Returns:
        points: torch.Tensor of shape (n, 2)
    """
    r1 = torch.rand(n, device=v0.device)
    r2 = torch.rand(n, device=v0.device)

    sqrt_r1 = torch.sqrt(r1)
    u = 1 - sqrt_r1
    v = r2 * sqrt_r1
    w = 1 - u - v

    # 重心坐标转换为实际坐标
    points = u[:, None] * v0 + v[:, None] * v1 + w[:, None] * v2
    return points

def SamplePointsFromTri_GPU(xys, tri, points_per_triangle=4):
    """
        Args:
            xys_tensor: torch.Tensor of shape (N, 2), must be on desired device
            tri: scipy.spatial.Delaunay object (tri.simplices will be used, still on CPU)
            points_per_triangle: int
        Returns:
            sampled_points: torch.Tensor of shape (T * points_per_triangle, 2)
            tri_ids: torch.Tensor of shape (T * points_per_triangle,)
        """
    device = xys.device
    time1 = time.time()

    sampled_points_list = []
    tri_ids_list = []

    simplices = tri.simplices  # shape: (T, 3)
    for tri_idx, simplex in enumerate(simplices):
        v0 = xys[simplex[0]]
        v1 = xys[simplex[1]]
        v2 = xys[simplex[2]]

        pts = sample_points_in_triangle_torch(v0, v1, v2, points_per_triangle)
        sampled_points_list.append(pts)
        tri_ids_list.append(torch.full((points_per_triangle,), tri_idx, device=device, dtype=torch.long))

    sampled_points = torch.cat(sampled_points_list, dim=0)
    tri_ids = torch.cat(tri_ids_list, dim=0)

    time2 = time.time()
    print(f"Sample Points: {time2 - time1:.3f}s, Sampled Point Number: {sampled_points.shape[0]}")

    return sampled_points, tri_ids

def sample_points_in_triangle(v0, v1, v2, n):
    """
    在一个三角形内均匀采样 n 个点（重心采样法）。
    """
    r1 = np.random.rand(n)
    r2 = np.random.rand(n)

    sqrt_r1 = np.sqrt(r1)
    u = 1 - sqrt_r1
    v = r2 * sqrt_r1
    w = 1 - u - v

    # 重心坐标 -> 实际点坐标
    points = u[:, None] * v0 + v[:, None] * v1 + w[:, None] * v2
    return points

def SamplePointsFromTri(xys, tri, points_per_triangle=4):
    Tri_IDs = []
    sampled_points = []

    for tri_idx, simplex in enumerate(tri.simplices):
        v0, v1, v2 = xys[simplex]
        pts = sample_points_in_triangle(v0, v1, v2, points_per_triangle)
        sampled_points.append(pts)
        for i in range(points_per_triangle):
            Tri_IDs.append(tri_idx)

    sampled_points = np.vstack(sampled_points)  # 合并成一个数组
    return sampled_points, Tri_IDs

def Get_3D_From_2D_GPU(point3D_ids : np.array, POINT3D_IDs, xyzs, rgbs):
    """
        将 2D 点对应的 point3D_ids 映射到 3D 空间坐标和 RGB 颜色。

        Args:
            point3D_ids: (N,) int tensor，2D 点对应的 3D 点 ID
            POINT3D_IDs: (M,) int tensor，全部3D点的ID
            xyzs: (M, 3) float tensor，对应3D坐标
            rgbs: (M, 3) float tensor，对应颜色 (已归一化)

        Returns:
            corresponding_3d_points: (N_valid, 3) float tensor
            corresponding_3d_points_rgb: (N_valid, 3) float tensor
        """
    # 创建 id -> index 映射
    id_to_index = {int(pid[0]): idx for idx, pid in enumerate(POINT3D_IDs.tolist())}

    valid_indices = []
    for pid in point3D_ids.tolist():
        if pid in id_to_index:
            valid_indices.append(id_to_index[pid])

    if len(valid_indices) == 0:
        print("未找到任何匹配的 3D 点 ID!!!")
        return torch.empty((0, 3)), torch.empty((0, 3))

    valid_indices = torch.tensor(valid_indices, dtype=torch.long, device=xyzs.device)

    corresponding_3d_points = xyzs[valid_indices]
    corresponding_3d_points_rgb = rgbs[valid_indices]

    print(f"找到 {corresponding_3d_points.shape[0]} 个 2D 点成功映射到 3D 点。")

    return corresponding_3d_points, corresponding_3d_points_rgb

def Get_3D_From_2D(point3D_ids : np.array, POINT3D_IDs : np.array, xyzs : np.array, rgbs : np.array):
    time1 = time.time()
    # 步骤 1：创建 ID 到坐标的映射字典
    id_to_xyz = {int(pid): [xyz, rgb] for pid, xyz, rgb in zip(POINT3D_IDs, xyzs, rgbs)}
    print(f"{time.time() - time1}s")

    # 步骤 2：找出所有2D点对应的3D坐标（跳过找不到ID的点）
    corresponding_3d_points = np.array([
        id_to_xyz[pid][0] for pid in point3D_ids if pid in id_to_xyz
    ])
    corresponding_3d_points_rgb = np.array([
        id_to_xyz[pid][1] for pid in point3D_ids if pid in id_to_xyz
    ])

    return corresponding_3d_points, corresponding_3d_points_rgb

def Get_3D_From_2D_2(point3D_ids: np.array, POINT3D_IDs, xyzs, rgbs):
    time1 = time.time()
    # 步骤 1：创建 ID 到坐标的映射字典
    id_to_xyz = {int(pid): [xyz, rgb] for pid, xyz, rgb in zip(POINT3D_IDs, xyzs, rgbs)}
    print(f"{time.time() - time1}s")

    time1 = time.time()
    corresponding_3d_points = []
    corresponding_3d_points_rgb = []
    for pid in point3D_ids:
        if pid in id_to_xyz:
            corresponding_3d_points.append(id_to_xyz[pid][0])
            corresponding_3d_points_rgb.append(id_to_xyz[pid][1])

    corresponding_3d_points = np.array(corresponding_3d_points)
    corresponding_3d_points_rgb = np.array(corresponding_3d_points_rgb)
    print(f"{time.time() - time1}s")

    return corresponding_3d_points, corresponding_3d_points_rgb

def Get_3D_From_2D_3(point3D_ids: np.array, sampled_point3D_ids: np.array, POINT3D_IDs, xyzs, rgbs):
    # 步骤 1：创建 ID 到坐标的映射字典
    id_to_xyz = {int(pid): [xyz, rgb] for pid, xyz, rgb in zip(POINT3D_IDs, xyzs, rgbs)}

    corresponding_3d_points = []
    sampled_corresponding_3d_points = []
    corresponding_3d_points_rgb = []
    sampled_corresponding_3d_points_rgb = []
    for pid in point3D_ids:
        if pid in id_to_xyz:
            corresponding_3d_points.append(id_to_xyz[pid][0])
            corresponding_3d_points_rgb.append(id_to_xyz[pid][1])

    for pid in sampled_point3D_ids:
        if pid in id_to_xyz:
            sampled_corresponding_3d_points.append(id_to_xyz[pid][0])
            sampled_corresponding_3d_points_rgb.append(id_to_xyz[pid][1])

    corresponding_3d_points = np.array(corresponding_3d_points)
    corresponding_3d_points_rgb = np.array(corresponding_3d_points_rgb)
    sampled_corresponding_3d_points = np.array(sampled_corresponding_3d_points)
    sampled_corresponding_3d_points_rgb = np.array(sampled_corresponding_3d_points_rgb)

    return corresponding_3d_points, corresponding_3d_points_rgb, sampled_corresponding_3d_points, sampled_corresponding_3d_points_rgb

def Get_Tri_Mask(extr, width, height, Get3D_ID=False, device='cuda', Silence=False, No_Key_Region=False):
    time1 = time.time()

    '''
    point3D_ids = []
    xys = []

    for i in range(len(extr.point3D_ids)):
        if extr.point3D_ids[i] != -1:
            point3D_ids.append(extr.point3D_ids[i])
            xys.append(extr.xys[i])

    # point3D_ids_tensor = torch.tensor(point3D_ids, dtype=torch.long, device=device)
    # xys_tensor = torch.tensor(xys, dtype=torch.float32, device=device)
    xys = np.array(xys)
    point3D_ids = np.array(point3D_ids)'''

    '''
    # 首先统计有多少个2D点对应的3D点ID不是-1
    Have_3D_Num = 0
    IDs = []
    for i in range(len(extr.point3D_ids)):
        if extr.point3D_ids[i] != -1:
            Have_3D_Num = Have_3D_Num + 1
            IDs.append(i)

    # 初始化np.array
    point3D_ids = np.empty(Have_3D_Num)
    xys = np.empty((Have_3D_Num, 2))

    for i in range(len(IDs)):
        point3D_ids[i] = extr.point3D_ids[IDs[i]]
        xys[i][0] = extr.xys[IDs[i]][0]
        xys[i][1] = extr.xys[IDs[i]][1]'''

    valid_mask = extr.point3D_ids != -1
    xys = extr.xys[valid_mask]
    point3D_ids = extr.point3D_ids[valid_mask]

    # 如果当前影像没有3维点（异常），则直接返回None
    if xys.shape[0] <= 0:
        if not Get3D_ID:
            return torch.ones(2, dtype=torch.bool).cuda()
        else:
            # 后续可用 GPU 加速的数据（以 torch.Tensor 返回）
            # point3D_ids_tensor = torch.from_numpy(point3D_ids).to(device)
            # xys_tensor = torch.from_numpy(xys).float().to(device)
            # point3D_ids_tensor = torch.tensor(point3D_ids, dtype=torch.int32, device=device)
            # xys_tensor = torch.tensor(xys, device=device)

            # return downsampled_mask.unsqueeze(0), point3D_ids_tensor, xys_tensor, tri
            return torch.ones(2, dtype=torch.bool).cuda(), None, xys, None

    # 构建 Delaunay 三角剖分
    tri = Delaunay(xys)

    if not No_Key_Region:
        mask = np.zeros((height, width), dtype=np.uint8)

        # 绘制每个三角形
        for simplex in tri.simplices:
            pts = xys[simplex].astype(np.int32)
            cv2.fillConvexPoly(mask, pts, 255)
    else:
        mask = np.ones((height, width), dtype=np.uint8)

    # 转为 PyTorch tensor 并降采样
    mask = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    if width > 1600:
        target_width = 1600  # 你希望的目标宽度
        target_height = int(height * target_width / width)  # 你希望的目标高度
        downsampled_mask = F.interpolate(mask, size=(target_height, target_width), mode='nearest')
        downsampled_mask = downsampled_mask.squeeze().bool()
        downsampled_mask = downsampled_mask
    else:
        downsampled_mask = mask.squeeze().bool()

    time2 = time.time()
    if not Silence:
        print(f"\nDelaunay Generation: {time2 - time1}s, 2D Points Number: {len(xys)}, Triangle Number: {tri.simplices.shape[0]}")

    if not Get3D_ID:
        return downsampled_mask.unsqueeze(0)
    else:
        # 后续可用 GPU 加速的数据（以 torch.Tensor 返回）
        # point3D_ids_tensor = torch.from_numpy(point3D_ids).to(device)
        # xys_tensor = torch.from_numpy(xys).float().to(device)
        # point3D_ids_tensor = torch.tensor(point3D_ids, dtype=torch.int32, device=device)
        # xys_tensor = torch.tensor(xys, device=device)

        # return downsampled_mask.unsqueeze(0), point3D_ids_tensor, xys_tensor, tri
        return downsampled_mask.unsqueeze(0), point3D_ids, xys, tri

def PointsFilter(sampled_points, mask, H_orig, W_orig, H_down, W_down, Tri_IDs=[], point3D_ids=None):
    x_down = (sampled_points[:, 0] * W_down / W_orig).astype(int)
    y_down = (sampled_points[:, 1] * H_down / H_orig).astype(int)

    downsampled_pixel_indices = y_down * W_down + x_down

    mask_flat = mask.flatten()

    valid_mask = mask_flat[downsampled_pixel_indices]

    # 筛选原始二维点
    selected_points = sampled_points[valid_mask]

    if Tri_IDs != []:
        Tri_IDs = np.array(Tri_IDs)
        Tri_IDs = Tri_IDs[valid_mask]
        Tri_IDs = list(Tri_IDs)

    if point3D_ids is not None:
        point3D_ids = point3D_ids[valid_mask]

    if Tri_IDs != []:
        return selected_points, Tri_IDs
    else:
        return selected_points, point3D_ids

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, Do_Get_Tri_Mask=False, No_Key_Region=False):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}: {}/{}".format(image_name, idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        if os.path.exists(image_path):
            image = Image.open(image_path)
        elif os.path.exists(image_path.split(".")[0] + ".jpg"):
            image_path = image_path.split(".")[0] + ".jpg"
            image = Image.open(image_path)
        else:
            image_path = image_path.split(".")[0] + ".png"
            image = Image.open(image_path)

        Tri_Mask = None
        if Do_Get_Tri_Mask:
            Tri_Mask = Get_Tri_Mask(extr, width, height, No_Key_Region=No_Key_Region)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, Tri_Mask=Tri_Mask)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def On_the_Fly_readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, CurrentImagesNames, Do_Get_Tri_Mask=False):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        if image_name not in CurrentImagesNames:
            if os.path.exists(image_path):
                image = Image.open(image_path)
            elif os.path.exists(image_path.split(".")[0] + ".jpg"):
                image_path = image_path.split(".")[0] + ".jpg"
                image = Image.open(image_path)
            else:
                image_path = image_path.split(".")[0] + ".png"
                image = Image.open(image_path)

            if Do_Get_Tri_Mask:
                Tri_Mask = Get_Tri_Mask(extr, width, height)
            else:
                Tri_Mask = None
        else:
            image = None
            Tri_Mask = None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, Tri_Mask=Tri_Mask)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def SingleImage_readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, Do_Get_Tri_Mask=False, Image_Name="", CurrentImagesNames=[], Silence=False, No_Key_Region=False):
    cam_infos = []
    point3D_ids_list = {}
    xys_list = {}
    tri_list = {}
    for idx, key in enumerate(cam_extrinsics):
        if not Silence:
            sys.stdout.write('\r')
            # the exact output you're looking for:
            sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
            sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        if (image_name != "" and image_name == Image_Name.split(".")[0]) or (CurrentImagesNames != [] and image_name not in CurrentImagesNames):
            if os.path.exists(image_path):
                image_path = image_path
            elif os.path.exists(image_path.split(".")[0] + ".jpg"):
                image_path = image_path.split(".")[0] + ".jpg"
            else:
                image_path = image_path.split(".")[0] + ".png"
            # print(image_path)

            image = Image.open(image_path)
            # print(image_path)

            if Do_Get_Tri_Mask:
                Tri_Mask, point3D_ids, xys, tri = Get_Tri_Mask(extr, width, height, True, Silence=Silence, No_Key_Region=No_Key_Region)
                point3D_ids_list[image_name] = point3D_ids
                xys_list[image_name] = xys
                tri_list[image_name] = tri
            else:
                Tri_Mask = None
        else:
            image = None
            Tri_Mask = None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                  image_path=image_path, image_name=image_name, width=width, height=height,
                                  Tri_Mask=Tri_Mask)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')

    # return cam_infos, point3D_ids, xys, tri
    return cam_infos, point3D_ids_list, xys_list, tri_list

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb, get_ply=True):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])

    if get_ply:
        ply_data.write(path)
    else:
        return ply_data

def On_the_Fly_readColmapSceneInfo(path, images, eval, CurrentImagesNames, llffhold=8, get_ply=True, Do_Get_Tri_Mask=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = On_the_Fly_readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), CurrentImagesNames=CurrentImagesNames, Do_Get_Tri_Mask=Do_Get_Tri_Mask)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)

        if get_ply:
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            storePly(ply_path, xyz, rgb)
        else:
            plydata = storePly(ply_path, xyz, rgb, False)
            vertices = plydata['vertex']
            positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
            colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
            normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
            pcd = BasicPointCloud(points=positions, colors=colors, normals=normals)
            ply_path = None

    if get_ply:
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def SingleImage_readColmapSceneInfo_Part1(path, images, eval, llffhold=8, Image_Name="", CurrentImagesNames=[], Do_Get_Tri_Mask=False, Diary=None, Silence=False, No_Key_Region=False):
    time1 = time.time()
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    time2 = time.time()
    if not Silence:
        print(f"Read in extrinsic and intrinsic file: {time2 - time1}s")
    Diary.write(f"Read in extrinsic and intrinsic file: {time2 - time1}s\n")

    time1 = time.time()
    reading_dir = "images" if images == None else images
    image_folder_path = os.path.join(os.path.dirname(path), reading_dir)
    if not os.path.exists(image_folder_path):
        image_folder_path = os.path.join(path, reading_dir)
    cam_infos_unsorted, point3D_ids_list, xys_list, tri_list = SingleImage_readColmapCameras(cam_extrinsics=cam_extrinsics,
                                                                              cam_intrinsics=cam_intrinsics,
                                                                              images_folder=image_folder_path,
                                                                              Image_Name=Image_Name,
                                                                              Do_Get_Tri_Mask=Do_Get_Tri_Mask,
                                                                              CurrentImagesNames=CurrentImagesNames,
                                                                              Silence=Silence,
                                                                              No_Key_Region=No_Key_Region)

    time2 = time.time()
    if not Silence:
        print(f"read Colmap Cameras: {time2 - time1}s")
    Diary.write(f"read Colmap Cameras: {time2 - time1}s\n")

    time1 = time.time()
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # 仅保留选中的影像（这一步在渐进式训练中不进行，因为需要更新其他训练影像的位姿）
    if CurrentImagesNames == []:
        for i in range(len(train_cam_infos)):
            if Image_Name != "" and train_cam_infos[i].image_name == Image_Name.split('.')[0]:
                train_cam_infos = [train_cam_infos[i]]
                break

    time2 = time.time()
    if not Silence:
        print(f"After read Colmap Cameras: {time2 - time1}s")
    Diary.write(f"After read Colmap Cameras: {time2 - time1}s\n")
    return train_cam_infos, test_cam_infos, nerf_normalization, point3D_ids_list, xys_list, tri_list

def SingleImage_readColmapSceneInfo_Part2(path, train_cam_infos, test_cam_infos, nerf_normalization, point3D_ids_list, xys_list, tri_list, NewCams, OriginImageHeight, OriginImageWidth, points_per_triangle=4, device="cuda", get_ply=True, Diary=None, Silence=False):
    all_points_3d_Stack = []
    all_colors_Stack = []

    time1 = time.time()
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")

    if not os.path.exists(ply_path):
        try:
            POINT3D_IDs, xyz, rgb, _ = read_points3D_ID_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)

        time2 = time.time()
        if not Silence:
            print(f"read_points3D_ID_binary: {time2 - time1}s")
        Diary.write(f"read_points3D_ID_binary: {time2 - time1}s\n")

        for key in list(point3D_ids_list.keys()):
            # 依据三角网进行采样
            time1 = time.time()
            sampled_points, Tri_IDs = SamplePointsFromTri(xys_list[key], tri_list[key], points_per_triangle)
            time2 = time.time()
            if not Silence:
                print(f"Sample Points: {time2 - time1}s, Sampled Point Number: {sampled_points.shape[0]}")
            Diary.write(f"Sample Points: {time2 - time1}s, Sampled Point Number: {sampled_points.shape[0]}\n")

            # 进行稀疏点云过滤，仅保留部分需要的点
            time1 = time.time()
            LogMask = None
            print(len(NewCams), key)
            for i in range(len(NewCams)):
                if NewCams[i].image_name == key:
                    LogMask = NewCams[i].LogMask
                    break

            sampled_points, Tri_IDs = PointsFilter(sampled_points, LogMask, OriginImageHeight, OriginImageWidth, LogMask.shape[0], LogMask.shape[1], Tri_IDs=Tri_IDs)
            time2 = time.time()
            if not Silence:
                print(f"Points Filter for sampled_points: {time2 - time1}s, Sampled Points Number: {sampled_points.shape[0]}")
            Diary.write(f"Points Filter for sampled_points: {time2 - time1}s, Sampled Point Number: {sampled_points.shape[0]}\n")

            time1 = time.time()
            if xys_list[key].shape[0] > 500:
                sampled_xys, sampled_point3D_ids = PointsFilter(xys_list[key], LogMask, OriginImageHeight, OriginImageWidth,
                                                                LogMask.shape[0], LogMask.shape[1],
                                                                point3D_ids=point3D_ids_list[key])
            else:
                sampled_xys = xys_list[key]
                sampled_point3D_ids = point3D_ids_list[key]
            time2 = time.time()
            if not Silence:
                print(f"Points Filter for xys: {time2 - time1}s, Sampled Points Number: {sampled_xys.shape[0]}")
            Diary.write(f"Points Filter for xys: {time2 - time1}s, Sampled Point Number: {sampled_xys.shape[0]}\n")

            time1 = time.time()
            corresponding_3d_points, corresponding_3d_points_rgb, sampled_corresponding_3d_points, sampled_corresponding_3d_points_rgb = Get_3D_From_2D_3(point3D_ids_list[key], sampled_point3D_ids, POINT3D_IDs, xyz, rgb)
            time2 = time.time()
            if not Silence:
                print(f"Find {corresponding_3d_points.shape} + {sampled_corresponding_3d_points.shape} 2D pts to 3D pts => {time2 - time1}s")
            Diary.write(f"Find {corresponding_3d_points.shape} + {sampled_corresponding_3d_points.shape} 2D pts to 3D pts => {time2 - time1}s\n")

            # 获取用于初始化的稀疏点云
            if device == 'cuda':
                time1 = time.time()
                # GPU方法
                tri_simplices = torch.tensor(tri_list[key].simplices, dtype=torch.long, device='cuda')
                xys = torch.tensor(xys_list[key], device=device)

                if sampled_points.shape[0] > 0:
                    sampled_points = torch.tensor(sampled_points, dtype=torch.float32, device=device)
                    Tri_IDs = torch.tensor(Tri_IDs, dtype=torch.long, device=device)
                else:
                    sampled_points = None
                    Tri_IDs = None

                corresponding_3d_points = torch.tensor(corresponding_3d_points, dtype=torch.float32, device=device)
                corresponding_3d_points_rgb = torch.tensor(corresponding_3d_points_rgb, dtype=torch.float32, device=device)

                if sampled_xys.shape[0] > 0:
                    sampled_corresponding_3d_points = torch.tensor(sampled_corresponding_3d_points, dtype=torch.float32, device=device)
                    sampled_corresponding_3d_points_rgb = torch.tensor(sampled_corresponding_3d_points_rgb, dtype=torch.float32, device=device)
                else:
                    sampled_corresponding_3d_points = None
                    sampled_corresponding_3d_points_rgb = None

                time2 = time.time()
                if not Silence:
                    print(f"CPU_numpy to GPU_tensor => {time2 - time1}s")
                Diary.write(f"CPU_numpy to GPU_tensor => {time2 - time1}s\n")

                time1 = time.time()
                all_points_3d, all_colors = Get_3D_Sampled_Points_GPU(sampled_points, Tri_IDs, tri_simplices, xys,
                                                                      corresponding_3d_points,
                                                                      corresponding_3d_points_rgb,
                                                                      sampled_corresponding_3d_points,
                                                                      sampled_corresponding_3d_points_rgb)
                time2 = time.time()
                if Tri_IDs is not None:
                    if not Silence:
                        print(f"3D Sample Points (GPU): {time2 - time1}s {sampled_points.shape[0]} Sampled pts finish Interpolation")
                    Diary.write(f"3D Sample Points (GPU): {time2 - time1}s {sampled_points.shape[0]} Sampled pts finish Interpolation\n")
                else:
                    if not Silence:
                        print(f"3D Sample Points (GPU): {time2 - time1}s {0} Sampled pts finish Interpolation")
                    Diary.write(f"3D Sample Points (GPU): {time2 - time1}s {0} Sampled pts finish Interpolation\n")

                time1 = time.time()
                if all_points_3d is not None:
                    all_points_3d = all_points_3d.detach().cpu().numpy()
                    all_colors = all_colors.detach().cpu().numpy()
                    if len(list(point3D_ids_list.keys())) >= 2:
                        all_points_3d_Stack.append(all_points_3d)
                        all_colors_Stack.append(all_colors)
                time2 = time.time()
                if not Silence:
                    print(f"GPU_tensor to CPU_numpy: {time2 - time1}s")
                Diary.write(f"GPU_tensor to CPU_numpy: {time2 - time1}s\n")
            else:
                # CPU方法
                all_points_3d, all_colors = Get_3D_Sampled_Points(sampled_points, Tri_IDs, tri_list[key], xys_list[key], corresponding_3d_points, corresponding_3d_points_rgb, sampled_corresponding_3d_points, sampled_corresponding_3d_points_rgb)

        if len(list(point3D_ids_list.keys())) >= 2:
            all_points_3d = np.concatenate(all_points_3d_Stack, axis=0)
            all_colors = np.concatenate(all_colors_Stack, axis=0)

        time1 = time.time()
        if all_points_3d is not None:
            print(f"Next Newly Added Gaussians Num: {all_points_3d.shape[0]}")
            if get_ply:
                if not Silence:
                    print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
                # storePly(ply_path, xyz, rgb)
                storePly(ply_path, all_points_3d, all_colors)
            else:
                plydata = storePly(ply_path, all_points_3d, all_colors, False)
                vertices = plydata['vertex']
                positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
                colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
                normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
                pcd = BasicPointCloud(points=positions, colors=colors, normals=normals)
                ply_path = None
        else:
            pcd = None
            ply_path = None
        time2 = time.time()
        if not Silence:
            print(f"Get pcd if need: {time2 - time1}s")
        Diary.write(f"Get pcd if need: {time2 - time1}s\n")

    time1 = time.time()
    if get_ply:
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None
    time2 = time.time()
    if not Silence:
        print(f"Get ply if need: {time2 - time1}s")
    Diary.write(f"Get ply if need: {time2 - time1}s\n")

    time1 = time.time()
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    time2 = time.time()
    if not Silence:
        print(f"Get SceneInfo: {time2 - time1}s")
    Diary.write(f"SceneInfo ply: {time2 - time1}s\n")

    return scene_info

def SingleImage_readColmapSceneInfo(path, images, eval, llffhold=8, get_ply=True, Do_Get_Tri_Mask=False, Image_Name="", points_per_triangle=4, device="cuda", CurrentImagesNames=[]):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted, point3D_ids, xys, tri = SingleImage_readColmapCameras(cam_extrinsics=cam_extrinsics,
                                                                              cam_intrinsics=cam_intrinsics,
                                                                              images_folder=os.path.join(path, reading_dir),
                                                                              Image_Name=Image_Name,
                                                                              Do_Get_Tri_Mask=Do_Get_Tri_Mask,
                                                                              CurrentImagesNames=CurrentImagesNames)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # 仅保留选中的影像（这一步在渐进式训练中不进行，因为需要更新其他训练影像的位姿）
    if CurrentImagesNames == []:
        for i in range(len(train_cam_infos)):
            if Image_Name != "" and train_cam_infos[i].image_name == Image_Name.split('.')[0]:
                train_cam_infos = [train_cam_infos[i]]
                break

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")

    if not os.path.exists(ply_path):
        try:
            POINT3D_IDs, xyz, rgb, _ = read_points3D_ID_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)

        # 仅取出选中影像可见的全部3D点以及其颜色信息
        corresponding_3d_points, corresponding_3d_points_rgb = Get_3D_From_2D(point3D_ids, POINT3D_IDs, xyz, rgb)

        # 依据三角网进行采样
        sampled_points, Tri_IDs = SamplePointsFromTri(xys, tri, points_per_triangle)

        # 获取用于初始化的稀疏点云
        if device == 'cuda':
            # GPU方法
            tri_simplices = torch.tensor(tri.simplices, dtype=torch.long, device='cuda')
            Tri_IDs = torch.tensor(Tri_IDs, dtype=torch.long, device=device)
            xys = torch.tensor(xys, device=device)
            sampled_points = torch.tensor(sampled_points, dtype=torch.float32, device=device)
            corresponding_3d_points = torch.tensor(corresponding_3d_points, dtype=torch.float32, device=device)
            corresponding_3d_points_rgb = torch.tensor(corresponding_3d_points_rgb, dtype=torch.float32, device=device)

            all_points_3d, all_colors = Get_3D_Sampled_Points_GPU(sampled_points, Tri_IDs, tri_simplices, xys, corresponding_3d_points, corresponding_3d_points_rgb)

            all_points_3d = all_points_3d.detach().cpu().numpy()
            all_colors = all_colors.detach().cpu().numpy()
        else:
            # CPU方法
            all_points_3d, all_colors = Get_3D_Sampled_Points(sampled_points, Tri_IDs, tri, xys, corresponding_3d_points, corresponding_3d_points_rgb)

        if get_ply:
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            # storePly(ply_path, xyz, rgb)
            storePly(ply_path, all_points_3d, all_colors)
        else:
            plydata = storePly(ply_path, xyz, rgb, False)
            vertices = plydata['vertex']
            positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
            colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
            normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
            pcd = BasicPointCloud(points=positions, colors=colors, normals=normals)
            ply_path = None

    if get_ply:
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readColmapSceneInfo(path, images, eval, llffhold=8, get_ply=True, Do_Get_Tri_Mask=False, No_Key_Region=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    image_folder_path = os.path.join(os.path.dirname(path), reading_dir)
    if not os.path.exists(image_folder_path):
        image_folder_path = os.path.join(path, reading_dir)
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=image_folder_path, Do_Get_Tri_Mask=Do_Get_Tri_Mask, No_Key_Region=No_Key_Region)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)

        if get_ply:
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            storePly(ply_path, xyz, rgb)
        else:
            plydata = storePly(ply_path, xyz, rgb, False)
            vertices = plydata['vertex']
            positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
            colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
            normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
            pcd = BasicPointCloud(points=positions, colors=colors, normals=normals)
            ply_path = None

    if get_ply:
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "On_the_Fly": On_the_Fly_readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Single_Image" : SingleImage_readColmapSceneInfo,
    "Single_Image1" : SingleImage_readColmapSceneInfo_Part1,
    "Single_Image2" : SingleImage_readColmapSceneInfo_Part2
}