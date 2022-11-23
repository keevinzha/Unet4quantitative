# -*- coding: utf-8 -*-
# @Time    : 2022/11/21 02:01
# @Author  : keevinzha
# @File    : torch_utils.py


import torch

def load_match_dict(model, model_path):
    # TODO what?
    # load pretrained model
    # using single gpu model with multi gpu, load dict before warp with nn.DataParallel
    pretrain_dict = torch.load(model_path)
    model_dict = model.state_dict()
    pretrain_dict = {k.replace('.module', ''): v for k, v in pretrain_dict.items()}
    # 1. filter out unnecessary keys
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if
                     k in model_dict and v.shape == model_dict[k].shape}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrain_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)