import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import cv2
import torchmetrics

import matplotlib.pyplot as plt

import wandb

from sklearn.metrics import *
def metrics(yhat, y, prefix=''):
    """
    long type inputs torch or numpy
    """

    try:
        yhat = yhat.flatten().cpu().detach().numpy()
        y = y.flatten().cpu().detach().numpy()
    except:
        yhat = yhat.flatten().numpy()
        y = y.flatten().numpy()

    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    accuracy = (tp+tn)/(tn+fp+fn+tp)
    iou = tp/(tp+fp+fn)
    dice = 2*tp/(2*tp+fp+fn)
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)

    return {'specificity'+prefix:specificity,
            'sensitivity'+prefix:sensitivity,
            'dice'+prefix:dice,
            'iou'+prefix:iou,
            'accuracy'+prefix:accuracy}

def Activation(tensor,T=1,recon=True):
    if recon==True:
        return tensor
    elif tensor.shape[1] != 1:
        return F.softmax(tensor/T,1)
    else:
        return F.sigmoid(tensor/T)

import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

# 먼저 warm up을 위하여 optimizer에 입력되는 learning rate = 0 또는 0에 가까운 아주 작은 값을 입력합니다.
# 위 코드의 스케쥴러에서는 T_0, T_mult, eta_max 외에 T_up, gamma 값을 가집니다.
# T_0, T_mult의 사용법은 pytorch 공식 CosineAnnealingWarmUpRestarts와 동일합니다. eta_max는 learning rate의 최댓값을 뜻합니다. T_up은 Warm up 시 필요한 epoch 수를 지정하며 일반적으로 짧은 epoch 수를 지정합니다. gamma는 주기가 반복될수록 eta_max 곱해지는 스케일값 입니다.

# optimizer = optim.Adam(model.parameters(), lr = 0)
# scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=150, T_mult=1, eta_max=0.1,  T_up=10, gamma=0.5)