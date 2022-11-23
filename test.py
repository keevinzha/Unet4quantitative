# -*- coding: utf-8 -*-
# @Time    : 2022/11/22 16:44
# @Author  : keevinzha
# @File    : test.py

#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import torch
import pydicom

from options import get_train_args


if __name__ == '__main__':
    args = get_train_args()
    E1 = pydicom.dcmread('/Users/kevinguo/Documents/MTR_001/I225.dcm')
    E2 = pydicom.dcmread('/Users/kevinguo/Documents/MTR_001/I226.dcm')
    T2 = pydicom.dcmread('/Users/kevinguo/Documents/MTR_001/I433.dcm')
    E1_img = torch.from_numpy(E1.pixel_array)
    E2_img = torch.from_numpy(E2.pixel_array)
    T2_img = torch.from_numpy(T2.pixel_array)
    a = torch.randn(3, 2)
    b = torch.randn(3, 2)
    c = torch.stack((a,b), dim=0)
    imgs = torch.stack((E1_img, E2_img), dim=0)
    print('done')
