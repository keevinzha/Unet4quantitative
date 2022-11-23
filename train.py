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

from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from options import prepare_train_args
from utils.logger import Logger
from data.data_entry import select_train_loader, select_eval_loader
from model.model_entry import select_model
from utils.torch_utils import load_match_dict
from metrics import *

class Trainer:
    def __init__(self):
        args = prepare_train_args()
        self.args = args()
        torch.manual_seed(args.seed)
        self.logger = Logger(args)
        self.train_loader = select_train_loader(args)
        self.val_loader = select_eval_loader(args)

        self.model = select_model(args)
        if args.load_model_path != '':
            print("using pre-trained weights")
            if args.load_not_strict:
                # TODO what??
                load_match_dict(self.model, args.load_model_path)
            else:
                self.model.load_state_dict(torch.load(args.load_model_path).state_dict())

        self.model = torch.nn.DataParallel(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr,
                                          betas=(self.args.momentum, self.args.beta),
                                          weight_decay=self.args.weight_decay) # TODO args
    def train(self):
        for epoch in range(self.args.epochs):
            self.train_per_epoch(epoch)
            self.val_per_epoch(epoch)
            self.logger.save_curves(epoch)
            self.logger.save_check_point(self.model, epoch)
    def train_per_epoch(self, epoch):
        self.model.train()

        for i, data in enumerate(self.train_loader):
            img, pred, label = self.step(data)

            metrics = self.compute_metrics(pred, label, is_train=True)

            loss = metrics['l1_ssim'] # TODO fill with metrics_name
            self.optimizer.zero_grad() # set zero at every begging of an epoch
            loss.backward() # backward
            self.optimizer.step() # update args

            for key in metrics.keys():
                self.logger.record_scalar(key, metrics[key])

            # save img at first step
            #if i == len(self.train_loader) - 1:
            #    self.logger.save_imgs(self.gen_imgs_to_write(img, pred, label, True), epoch)

            # monitor training progress
            if i % self.args.print_freq == 0:
                print('Train: Epoch {} batch {} Loss {}'.format(epoch, i, loss))
    def val_per_epoch(self, epoch):
        self.model.eval()
        for i, data in enumerate(self.val_loader):
            img, pred, label = self.step(data)
            metrics = self.compute_metrics(pred, label, is_train=False)

            for key in metrics.keys():
                self.logger.record_scalar(key, metrics[key])

            #if i == len(self.val_loader) - 1:
            #    self.logger.save_imgs(self.gen_imgs_to_write(img, pred, label, False), epoch)

    def step(self, data):
        img, label = data
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        pred = self.model(img)
        return img, pred, label

    def compute_metrics(self, pred, gt, is_train):
        # using this function to replace compute_loss
        l1_ssim_loss = L1(pred, gt) + SSIM(pred, gt)
        prefix = 'train/' if is_train else 'val/'
        metrics = {
            prefix + 'l1_ssim': l1_ssim_loss
        }
        return metrics

    #def gen_imgs_to_write(self, img, pred, label, is_train):
        # TODO call functions in visualization


def main():
    trainer = Trainer()
    trainer.train()

if __name__ == '__main__':
    main()
