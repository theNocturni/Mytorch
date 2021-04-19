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

def ce(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    img_ce1 = img
    img_ce2 = black_tophat(img,star(7))
    img_ce3 = hessian(img,sigmas=.1,alpha=0.5,beta=.5,gamma=10)
#     img_ce3 = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), img.shape[0]/30), -4, 128)
    return np.stack([img_ce1/255.,img_ce2/255.,img_ce3],-1)

def clahe(img):
    temp = np.zeros_like(img)
    for idx in range(temp.shape[-1]):
        adaptive_hist_range = True
#         adaptive_hist_range = False
#         temp[...,idx] = mclahe.mclahe(img[...,idx],kernel_size=(7,7),adaptive_hist_range=adaptive_hist_range)
        temp[...,idx] = mclahe.mclahe(img[...,idx],kernel_size=(15,15),adaptive_hist_range=adaptive_hist_range)
    return temp

class dataset():
    
    def __init__(self,data_root='dataset',dataset_type='train',transform=None):
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
        
        self.transform = transform
        print('total counts of dataset x {}, y {}'.format(len(self.x_list),len(self.y_list)))
        
    def __len__(self):
        return len(self.x_list)
  
    def __getitem__(self, idx):
#         x = imageio.imread(self.x_list[idx])    
#         y = imageio.imread(self.y_list[idx])
        x = cv2.imread(self.x_list[idx])
        y = cv2.imread(self.y_list[idx])
#         x_origin = x.copy()
        fname = self.x_list[idx]
        if len(y.shape)==2:
            y = np.expand_dims(y,-1)
        y[y!=0] = 1
        
        if self.transform:
            sample = self.transform(image = x, mask = y)
            x, y= sample['image'], sample['mask']
        
        x = ce(x)
        x = clahe(x)
        x[...,1] = 1-x[...,1]

        x = np.moveaxis(x,-1,0).astype(np.float32)
        y = np.moveaxis(y,-1,0).astype(np.long)
        x = torch.tensor(x)
        
        if len(np.unique(y))!=1:            
            y = torch.tensor(y)
            y = y[0].unsqueeze(0)
        else:
            y = None
        
        return {'x':x,'y':y,'fname':fname}
    
import albumentations as albu
patch_size = (576,576)

def augmentation_train():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        
        albu.OneOf([
        albu.RandomBrightnessContrast(brightness_limit=(-0.5, 0.3), contrast_limit=(-0.3, 0.3), p=0.5),
        albu.RandomGamma(gamma_limit=(60,130), p=.5),
        ],p=0.5),
                
#         albu.OneOf([
#         albu.RandomFog(fog_coef_lower=0.1,fog_coef_upper=.2,alpha_coef=0.04,p=0.3),
#         ],p=0.1),
        
        albu.OneOf([
        albu.GaussNoise(var_limit=0.01, mean=0, p=0.5),
        albu.MultiplicativeNoise(multiplier=(0.98, 1.02), p=0.5),
        ],p=0.3),
        
        albu.OneOf([
        albu.ElasticTransform(border_mode=cv2.BORDER_CONSTANT,interpolation=cv2.INTER_CUBIC,alpha=1,sigma=50,alpha_affine=50, p=0.5),
        albu.GridDistortion(border_mode=cv2.BORDER_CONSTANT,interpolation=cv2.INTER_CUBIC,distort_limit=(-0.3,0.3),num_steps=5, p=0.5),
        albu.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT,interpolation=cv2.INTER_CUBIC,distort_limit=(-.05,.05),shift_limit=(-0.05,0.05), p=0.5),
        albu.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_CUBIC, shift_limit=0.2, scale_limit=(0, 0.25), rotate_limit=45, p=0.5),   
        ],p=0.5),
        
#         albu.OneOf([            
# #         albu.IAASharpen(alpha=(0.1,0.4), lightness=(0.5, 0.), p=0.2),
#         albu.GaussianBlur(blur_limit=(3,5), p=0.5),
#         ],p=0.5),
             
        albu.PadIfNeeded(585, 585, border_mode=cv2.BORDER_CONSTANT, value=0,always_apply=True),
        albu.Resize(585, 585, interpolation=cv2.INTER_CUBIC, always_apply=True),
        albu.RandomCrop(height=patch_size[0], width=patch_size[1], always_apply=True),
    ]
    return albu.Compose(train_transform)

def augmentation_valid():
    test_transform = [
#         albu.Resize(height=1024, width=1024, always_apply=True),     
        albu.PadIfNeeded(585, 585, border_mode=cv2.BORDER_CONSTANT, value=0,always_apply=True),
        albu.CenterCrop(576, 576, always_apply=True)
    ]
    return albu.Compose(test_transform)
