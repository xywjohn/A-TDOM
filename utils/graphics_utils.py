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
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def getWorld2View2_TDOM(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    # scale = 0.14
    scale = 0.13
    S = np.array([
        [scale, 0, 0, 0],
        [0, scale, 0, 0],
        [0, 0, scale, 0],
        [0, 0, 0, 1]
    ])
    # 将 Rt 矩阵与缩放矩阵 S 相乘
    Rt = np.dot(S, Rt)

    return np.float32(Rt)


def getProjectionMatrix_TDOM(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    # zfar/100不会出现伪影啊！！！
    top = tanHalfFovY * zfar/100
    bottom = -top
    right = tanHalfFovX * zfar/100
    left = -right

    # top = tanHalfFovY * znear
    # bottom = -top
    # right = tanHalfFovX * znear
    # left = -right

    # print("top:", top)
    # print("right:", right)

    P = torch.zeros(4, 4)

    z_sign = 1.0


    # left = -4
    # right = 4
    # bottom = -4
    # top = 4
    # znear = 9999
    # zfar = 100000.0

    # 正射投影矩陣
    P[0, 0] = 2.0 / (right - left)
    P[0, 3] = - (right + left) / (right - left)
    P[1, 1] = 2.0 / (top - bottom)
    P[1, 3] = - (top + bottom) / (top - bottom)
    P[2, 2] = -2.0 / (zfar - znear)
    P[2, 3] = - (zfar + znear)/(zfar - znear)
    P[3, 3] = z_sign
    #
    # P[0, 0] = 1.0
    # P[0, 3] = - (right + left) / (right - left)
    # P[1, 1] = 1.0
    # P[1, 3] = - (top + bottom) / (top - bottom)
    # P[2, 2] = -1.0
    # P[2, 3] = - (zfar + znear)/(zfar - znear)
    # P[3, 3] = z_sign

    # P[0, 0] = 2.0 * znear / (right - left)
    # P[1, 1] = 2.0 * znear / (top - bottom)
    # P[0, 2] = (right + left) / (right - left)
    # P[1, 2] = (top + bottom) / (top - bottom)
    # P[3, 2] = z_sign
    # P[2, 2] = z_sign * zfar / (zfar - znear)
    # P[2, 3] = -(zfar * znear) / (zfar - znear)

    # P[0, 0] = 2 * fx / W
    # P[1, 1] = 2 * fy / H
    # P[0, 2] = 2 * (cx / W) - 0.5
    # P[1, 2] = 2 * (cy / H) - 0.5
    # P[2, 2] = -(zfar + znear) / (zfar - znear)
    # P[3, 2] = 1.0
    # P[2, 3] = -(2 * zfar * znear) / (zfar - znear)
    #



    return P