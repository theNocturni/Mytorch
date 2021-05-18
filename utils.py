import torch
import torch.nn
import argparse
import logging
from functools import wraps

import os
import torch
import torch.backends.cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision.utils import make_grid
import torchvision.transforms as transforms

import monai

import random, math, statistics
import numpy as np
from sklearn.metrics import *
# from kornia.morphology import top_hat
# import kornia

import wandb

# from utils import *
# from config import get_config
import utils
import models
import losses
import datasets

random_seed = 0
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
# set_determinism(seed=0)

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LEVEL = logging.DEBUG
logging.basicConfig(format=FORMAT, level=LEVEL)
log = logging.getLogger(__name__)
log.info('Entered module: %s' % __name__)


def logger(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        log = logging.getLogger(fn.__name__)
        log.info('Start running %s' % fn.__name__)

        out = fn(*args, **kwargs)

        log.info('Done running %s' % fn.__name__)
        # Return the return value
        return out

    return wrapper


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def weight_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
            
            
from skimage.morphology import white_tophat,black_tophat
from skimage.morphology import disk, star, square
import mclahe
import matplotlib.pyplot as plt

def show_samples(batch_x, batch_y, batch_yhat, message=''):
    '''
    all inputs should be shaped in BxCxHxW. (only for 2D segmentation)
    If prediction shape channel more than 2, you need to argmax it. (fixed)
    The first element of the batch will shown.
    '''
    if len(batch_yhat.shape)==4 and batch_yhat.shape[1]>1:
        batch_yhat = batch_yhat[:,1].unsqueeze(1)
#     elif len(batch_yhat.shape)==4 and batch_yhat.shape[1]==1:
#         batch_yhat = batch_yhat.round()        
    if len(batch_yhat.shape)==3:
        try:
            batch_yhat = batch_yhat.unsqueeze(1)
        except:
            batch_yhat = np.expand_dims(batch_yhat,1)
            
    idx= 0 
    plt.figure(figsize=(24,16))
    plt.subplot(151)
    plt.title(str(message)+'_x')
    plt.imshow(batch_x[idx,0].cpu().detach(),cmap='gray')
    plt.subplot(152)
    plt.title(str(message)+'_x')
    plt.imshow(batch_x[idx,1].cpu().detach(),cmap='gray')
    plt.subplot(153)
    plt.title(str(message)+'_x')
    plt.imshow(batch_x[idx,2].cpu().detach(),cmap='gray')
    plt.subplot(154)
    plt.title(str(message)+'_y')
    print(torch.unique(batch_y))
    plt.imshow(batch_yhat[idx,0].cpu().detach(),cmap='gray')
    batch_yhat = batch_yhat.round()        
    plt.subplot(155)
    plt.title(str(message)+'_yhat')
    plt.title(str(message)+'_y-yhat (Green:FP, Red:FN, White:TP)')
    
    temp = np.zeros((batch_x[idx,0].shape[0],batch_x[idx,0].shape[1],3))
    for idx_ in range(3):
        temp[...,idx_] = batch_y[idx,0].cpu().detach() # White (gt)
    try:
        diff = batch_y[idx,0].float().cpu().detach().numpy()-batch_yhat[idx,0].float().cpu().detach().numpy()
    except:
        diff = batch_y[idx,0]-batch_yhat[idx,0]    
    
    diff_fp = diff.copy()
    diff_fp[diff_fp!=-1] = 0
    diff_fp[diff_fp!=0] = 1
    diff_fn = diff.copy()
    diff_fn[diff_fn!=1] = 0
    diff_fn[diff_fn!=0] = 1
    
    temp[...,1] -= diff_fn #R   gt-fn
    temp[...,2] -= diff_fn #R   gt-fn
    temp[...,1] += diff_fp #G   gt+fp
    temp[temp!=0]=1
    
    plt.imshow(temp,alpha=1,cmap='gray')
    plt.show()
#     f1 =  f1_score(batch_y.cpu().detach().flatten(),batch_yhat.cpu().detach().flatten())
#     print('dice',f1)
#     return f1

from sklearn.metrics import *
def metrics(yhat,y):
    """
    Binary classification metric
    
    input : long type inputs torch or numpy
    output : various metric in dictionary form
    """
    
    try:
        try:
            yhat = yhat.flatten().cpu().detach().numpy()
            y = y.flatten().cpu().detach().numpy()
        except:
            yhat = yhat.flatten().numpy()
            y = y.flatten().numpy()
    except:
        yhat = yhat.flatten()
        y = y.flatten()
    
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    accuracy = (tp+tn)/(tn+fp+fn+tp)
    iou = tp/(tp+fp+fn)
    dice = 2*tp/(2*tp+fp+fn)
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    ppv = tp / (tp+fp)
    npv = tn / (tn+fn)
    
    return {'accuracy':accuracy,
            'dice':dice, 
            'iou':iou, 
            'npv':npv,
            'sensitivity':sensitivity,
            'specificity':specificity,
            'ppv':ppv,
           }