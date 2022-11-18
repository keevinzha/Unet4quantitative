# -*- coding: utf-8 -*-
# @Time    : 2022/10/27 10:01
# @Author  : keevinzha
# @File    : train.py
import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet.unet_model import UNet