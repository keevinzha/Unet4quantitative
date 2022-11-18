# -*- coding: utf-8 -*-
# @Time    : 2022/10/27 16:43
# @Author  : keevinzha
# @File    : dice_score.py
import torch
from torch import Tensor

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
