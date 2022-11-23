# -*- coding: utf-8 -*-
# @Time    : 2022/11/22 10:05
# @Author  : keevinzha
# @File    : qdess_dataset.py
#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import pydicom
import torch
import matplotlib.pyplot as plt
from os import path
from torch.utils.data import Dataset

from options import get_train_args



class QdessDataset(Dataset):
    def __init__(self, args, is_train):
        # super(self, QdessDataset).__init__()
        if is_train:
            data_qdess = args.train_qdess
        else:
            data_qdess = args.val_qdess
        infos = [line.split() for line in open(data_qdess).readlines()]

        self.E1_paths = [info[0] for info in infos]
        self.E2_paths = [info[1] for info in infos]
        self.T2_paths = [info[2] for info in infos]

    def preprocess(self, img, label): # not used now
        return img, label

    def __len__(self):
        return len(self.E1_paths)

    def __getitem__(self, idx):
        E1 = pydicom.dcmread(self.E1_paths[idx]).pixel_array
        E2 = pydicom.dcmread(self.E2_paths[idx]).pixel_array
        T2 = pydicom.dcmread(self.T2_paths[idx]).pixel_array
        E1 = torch.from_numpy(E1)
        E2 = torch.from_numpy(E2)
        T2 = torch.from_numpy(T2)
        img = torch.stack((E1, E2), dim=0) # (echo, y, x)
        label = T2
        img, label = self.preprocess(img, label)
        return img, label


if __name__ == '__main__':
    args = get_train_args()
    test = QdessDataset(args, True)