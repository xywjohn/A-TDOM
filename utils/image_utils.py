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

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2, mask=None):
    if mask is None:
        mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))
    else:
        # 如果是2D图像，扩展成 [1, H, W]
        if img1.dim() == 2:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)

        # 广播 mask 到图像通道
        mask = mask.expand_as(img1)  # [C, H, W]

        # 计算掩膜下的平方误差
        diff_sq = (img1 - img2) ** 2
        masked_diff_sq = diff_sq[mask]

        if masked_diff_sq.numel() == 0:
            return float('nan')  # 掩膜为空，不计算

        mse = masked_diff_sq.mean()
        psnr = 20 * torch.log10(1.0 / (torch.sqrt(mse) + 1e-8))

        return psnr



