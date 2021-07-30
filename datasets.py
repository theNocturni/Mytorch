import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import os
import glob
from natsort import natsorted

import numpy as np
import torch

import mclahe
import imageio

import cv2
import mclahe
import imageio

from PIL import Image
from skimage.filters import frangi
from skimage.morphology import white_tophat,black_tophat
from skimage.morphology import disk, star, square
from skimage.filters import *

import kornia
from kornia.morphology import *
from kornia.enhance import *

import matplotlib.pyplot as plt
import albumentations as albu

# def ce(img):
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
#     img_ce1 = img
#     img_ce2 = black_tophat(img,star(11))
#     img_ce3 = hessian(img,sigmas=.1,alpha=0.5,beta=.5,gamma=10)
#     return np.stack([img_ce1/255.,img_ce2/255.,img_ce3],-1)

def clahe(img,adaptive_hist_range=False):
    """
    input 1 numpy shape image ( H x W x C)
    """
    temp = np.zeros_like(img)
    for idx in range(temp.shape[-1]):
        temp[...,idx] = mclahe.mclahe(img[...,idx],adaptive_hist_range=adaptive_hist_range)
    return temp

# Data Module
class dataset():
    
    def __init__(self,data_root='path/to/data',dataset_type='train', transform_spatial=None, transform=None):
        
        self.data_root = data_root
        if dataset_type =='train':
            self.x_list = natsorted(glob.glob(data_root+'/x_train/*'))
            self.y_list = natsorted(glob.glob(data_root+'/y_train/*'))
        elif dataset_type =='valid':
            self.x_list = natsorted(glob.glob(data_root+'/x_valid/*'))
            self.y_list = natsorted(glob.glob(data_root+'/y_valid/*'))
        elif dataset_type =='test':
            self.x_list = natsorted(glob.glob(data_root+'/x_test/*'))
            self.y_list = natsorted(glob.glob(data_root+'/y_test/*'))
        elif dataset_type =='etest':
            self.x_list = natsorted(glob.glob(data_root+'/x_etest/*'))
            self.y_list = natsorted(glob.glob(data_root+'/y_etest/*'))
            self.x_list = self.x_list[:1]
        
        self.transform = transform
        self.transform_spatial = transform_spatial
        print('total counts of dataset x {}, y {}'.format(len(self.x_list),len(self.y_list)))
        
    def __len__(self):
        return len(self.x_list)
  
    def __getitem__(self, idx):
        fname = self.x_list[idx]
        x = cv2.imread(self.x_list[idx])
        y = cv2.imread(self.y_list[idx])
        x = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
        y[y!=0] = 1
        if len(y.shape)==2:
            y = np.expand_dims(y,-1)
        elif len(y.shape)==3 and y.shape[-1]==3:
            y = np.expand_dims(y[...,0],-1)
            
        if self.transform:
            sample = self.transform(image = x, mask = y)
            x, y= sample['image'], sample['mask']        

        x = x.astype(np.float32)
        x = clahe(x)
        
        if self.transform_spatial:
            sample = self.transform_spatial(image = x, mask = y)
            x, y= sample['image'], sample['mask']        
        
        x = np.moveaxis(x,-1,0).astype(np.float32)
        y = np.moveaxis(y,-1,0).astype(np.float32)

        x = torch.tensor(x)
        
        y = torch.tensor(y)
        y = y[0].unsqueeze(0)
        
        return {'x':x,'y':y,'fname':fname}

# class dataset_kornia():
    
#     def __init__(self,data_root='dataset',dataset_type='train', transform_crop=None, transform=None):
#         self.data_root = data_root
#         if dataset_type =='train':
#             self.x_list = natsorted(glob.glob(data_root+'/x_train/*'))
#             self.y_list = natsorted(glob.glob(data_root+'/y_train/*'))
#         elif dataset_type =='valid':
#             self.x_list = natsorted(glob.glob(data_root+'/x_valid/*'))
#             self.y_list = natsorted(glob.glob(data_root+'/y_valid/*'))
#         elif dataset_type =='test':
#             self.x_list = natsorted(glob.glob(data_root+'/x_test/*'))
#             self.y_list = natsorted(glob.glob(data_root+'/y_test/*'))
#         elif dataset_type =='etest':
#             self.x_list = natsorted(glob.glob(data_root+'/x_etest/*'))
#             self.y_list = natsorted(glob.glob(data_root+'/y_etest/*'))
#             self.x_list = self.x_list[:1]
            
#         self.transform = transform
#         print('total counts of dataset x {}, y {}'.format(len(self.x_list),len(self.y_list)))
        
#     def __len__(self):
#         return len(self.x_list)
  
#     def __getitem__(self, idx):
#         fname = self.x_list[idx]
#         x = cv2.imread(self.x_list[idx])
#         y = cv2.imread(self.y_list[idx])
#         x = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
        
#         if len(y.shape)==2:
#             y = np.expand_dims(y,-1)
            
        
#         if self.transform:
#             sample = self.transform(image = x, mask = y)
#             x, y= sample['image'], sample['mask']        

#         x = x.astype(np.float32)
#         x = clahe(x)
        
#         if self.transform_crop:
#             sample = self.transform_crop(image = x, mask = y)
#             x, y= sample['image'], sample['mask']        
        
#         x = np.moveaxis(x,-1,0).astype(np.float32)
#         y = np.moveaxis(y,-1,0).astype(np.float32)

#         x = torch.tensor(x)
        
#         y = torch.tensor(y)
#         y = y[0].unsqueeze(0)
        
#         kernel = torch.ones(13,13)
# #         kernel = torch.tensor(star(13)).float()
#         x = kornia.morphology.bottom_hat(x.unsqueeze(0), kernel)
#         x = kornia.enhance.normalize_min_max(x)
#         x = x.squeeze(0)
        
#         return {'x':x,'y':y,'fname':fname}

# augmentation
def augmentation_imagesize(data_padsize=None, data_cropsize=None, data_resize=None, data_patchsize = None):
    """
    sizes should be in 
    """
    transform = list()
    
    if data_padsize:
        if len(data_padsize.split('_'))==1:
            data_padsize = int(data_padsize)
            transform.append(albu.PadIfNeeded(data_padsize, data_padsize, border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=True))
        else:
            data_padsize_h = int(data_padsize.split('_')[0])
            data_padsize_w = int(data_padsize.split('_')[1])
            transform.append(albu.PadIfNeeded(data_padsize_h, data_padsize_w, border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=True))
    if data_cropsize:
        if len(data_cropsize.split('_'))==1:
            data_cropsize = int(data_cropsize)
            transform.append(albu.CenterCrop(data_cropsize, data_cropsize, always_apply=True))
        else:
            data_cropsize_h = int(data_cropsize.split('_')[0])
            data_cropsize_w = int(data_cropsize.split('_')[1])
            transform.append(albu.CenterCrop(data_cropsize_h, data_cropsize_w, always_apply=True))
    if data_resize:
        if len(data_resize.split('_'))==1:
            data_resize = int(data_resize)
            transform.append(albu.Resize(data_resize, data_resize, interpolation=cv2.INTER_CUBIC, always_apply=True))
        else:
            data_resize_h = int(data_resize.split('_')[0])
            data_resize_w = int(data_resize.split('_')[1])
            transform.append(albu.Resize(data_resize_h, data_resize_w, interpolation=cv2.INTER_CUBIC, always_apply=True))
    if data_patchsize:
        if len(data_patchsize.split('_'))==1:
            data_patchsize = int(data_patchsize)
            transform.append(albu.RandomCrop(data_patchsize, data_patchsize, always_apply=True))
        else:
            data_patchsize_h = int(data_patchsize.split('_')[0])
            data_patchsize_w = int(data_patchsize.split('_')[1])
            transform.append(albu.RandomCrop(height=data_patchsize_h, width=data_patchsize_w, always_apply=True))
            
    print(transform)

    return albu.Compose(transform)

def augmentation_train():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        
        albu.OneOf([
        albu.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), brightness_by_max=False, p=0.5),
        albu.RandomGamma(gamma_limit=(70,120), p=.5),
        albu.RandomToneCurve(scale=0.2,p=.5) 
        ],p=0.5),
                
        albu.OneOf([
        albu.RandomFog(fog_coef_lower=0.1, fog_coef_upper=.3, alpha_coef=0.04, p=0.3),
        albu.MotionBlur(blur_limit=3, p=0.3),
        albu.MedianBlur(blur_limit=3, p=0.3),
        albu.GlassBlur(sigma=0.1, max_delta=2, p=0.3), 
        ],p=0.1),
        
        albu.OneOf([
        albu.GaussNoise(var_limit=0.02, mean=0, p=0.5),
        albu.MultiplicativeNoise(multiplier=(0.98, 1.02), p=0.5),
        albu.ISONoise(color_shift=(0.01, 0.03),intensity=(0.1, 0.3),p=0.5),
        ],p=0.3),
        
        albu.OneOf([
        albu.ElasticTransform(border_mode=cv2.BORDER_CONSTANT,interpolation=cv2.INTER_CUBIC,alpha=1,sigma=50,alpha_affine=50, p=0.5),
        albu.GridDistortion(border_mode=cv2.BORDER_CONSTANT,interpolation=cv2.INTER_CUBIC,distort_limit=(-0.3,0.3),num_steps=5, p=0.5),
        albu.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT,interpolation=cv2.INTER_CUBIC,distort_limit=(-.05,.05),shift_limit=(-0.05,0.05), p=0.5),
        albu.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_CUBIC, shift_limit=(0.05,0.02), scale_limit=(-.1, 0), rotate_limit=2, p=0.5),   
        ],p=0.5),
                     
    ]
    return albu.Compose(train_transform)

def augmentation_valid():
    test_transform = [
        
    ]
    return albu.Compose(test_transform)