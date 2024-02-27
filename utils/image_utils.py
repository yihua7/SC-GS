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

# NeRF-DS Alex LPIPS
import lpips as lpips_lib
loss_fn_alex = lpips_lib.LPIPS(net='alex')
loss_fn_alex.net.cuda()
loss_fn_alex.scaling_layer.cuda()
loss_fn_alex.lins.cuda()
def alex_lpips(image1, image2):
  image1 = image1 * 2 - 1
  image2 = image2 * 2 - 1
  lpips = loss_fn_alex(image1, image2)
  return lpips


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


from piq import ssim, LPIPS
lpips = LPIPS()
