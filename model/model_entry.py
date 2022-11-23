# -*- coding: utf-8 -*-
# @Time    : 2022/11/21 01:56
# @Author  : keevinzha
# @File    : model_entry.py


from model.base.unet import Unet
import torch.nn as nn

def select_model(args):
    type2model = {
        # TODO fill this
    }
    model = type2model[args.model_type]
    return model

def equip_mulit_gpu(model, args):
    model = nn.DataParallel(model, device_ids=args.gpus)
    return model