# -*- coding: utf-8 -*-
# @Time    : 2022/11/23 09:52
# @Author  : keevinzha
# @File    : metrics.py

import pytorch_ssim
import torch.nn as nn

# args should be Variables
def L1(input, target):
    L1_loss = nn.L1Loss()
    loss = L1_loss(input, target)
    return loss

def SSIM(input, target):
    ssim_loss = pytorch_ssim.SSIM()
    loss = ssim_loss(input, target)
    return loss
