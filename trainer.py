import warnings
warnings.filterwarnings("ignore")

import logging.config
logging.config.dictConfig({
    'version': 1,
    # Other configs ...
    'disable_existing_loggers': True
})

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

import random
import math
import statistics
import numpy as np
from sklearn.metrics import *

import wandb

from utils import logger, weight_init
from config import get_config
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

# @logger
def save_model(net, cfg):
    if torch.cuda.device_count() == 1:
        torch.save(net.state_dict(),cfg.experiment_name+'.pt')
    else:
        torch.save(net.module.state_dict(),cfg.experiment_name+'.pt')

# @logger
def load_optimizer(net, cfg):
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return optimizer

# @logger
def load_network(cfg):
    if cfg.net =='manet':
        import segmentation_models_pytorch as smp
        net = smp.MAnet('timm-efficientnet-b7', in_channels=cfg.net_inputch, classes=cfg.net_outputch)
    if cfg.net =='waveletunet':
        net= models.Waveletunet(in_channel=cfg.net_inputch,out_channel=cfg.net_outputch, c=32, norm=cfg.net_norm,mc_dropout=cfg.net_mcdropout)
        net.apply(weight_init)
    if cfg.net =='attunet':
        net = models.AttU_Net(img_ch=cfg.net_inputch,output_ch=cfg.net_outputch,norm=cfg.net_norm,mc_dropout=cfg.net_mcdropout)
    if cfg.net =='r2attunet':
        net = models.R2AttU_Net(img_ch=cfg.net_inputch,output_ch=cfg.net_outputch,t=2,norm=cfg.net_norm,mc_dropout=cfg.net_mcdropout,nnblock=cfg.net_nnblock)
    if cfg.net == 'tightwaveletnet':
        net = TightWaveletnet(in_channel=cfg.net_inputch, out_channel=cfg.net_outputch, c=32, norm=cfg.net_norm)
    if cfg.net == 'axialattnet':
        net = models.axial50l(in_channel=1,)
    else:
        net = models.AttU_Net(img_ch=cfg.net_inputch,output_ch=cfg.net_outputch,norm=cfg.net_norm,mc_dropout=cfg.net_mcdropout)
        
    if cfg.net_pretrained is not None:
        weight = torch.load(cfg.net_pretrained)
        try:
            net.load_state_dict(weight,strict=False)
            print("weight loaded", torch.cuda.device_count(), "GPUs!")
        except:
            print('loading weight failed')
        del weight
        
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    return net

# @logger
def load_lossfn(loss_fn):
    if loss_fn =='dicece':
        lossfn = losses.DiceCELoss()
    if loss_fn =='boundaryce':
        lossfn = losses.BoundaryCELoss()
    if loss_fn =='cldicece':
        lossfn = losses.clDiceCELoss()
    else:
        lossfn = losses.CrossEntropyLoss()
    return lossfn

def softmax_T(tensor,T=100):
    return F.softmax(tensor/T,1)

# wandb image
segmentation_classes = ['background', 'vessel']

def labels():
    l = {}
    for i, label in enumerate(segmentation_classes):
        l[i] = label
    return l

def wb_mask(x, yhat,y):
    x = x.cpu().detach().numpy()[0]
    y = y.cpu().detach().numpy()[0]
    yhat = torch.argmax(yhat,1).cpu().detach().numpy()[0]
    
    x = np.moveaxis(x,0,-1)
    y = y[0]
    return wandb.Image(x, masks={
    "prediction" : {"mask_data" : yhat, "class_labels" : labels()},
    "ground truth" : {"mask_data" : y, "class_labels" : labels()}})

def main(cfg):
    # -------------------------------------------------------------------
    print(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # -------------------------------------------------------------------

    # optional: log model topology
    # -------------------------------------------------------------------
    # load data
    trainset = datasets.dataset(cfg.data_path, 'train', transform=datasets.augmentation_train())
    validset = datasets.dataset(cfg.data_path, 'valid', transform=datasets.augmentation_valid())
    testset = datasets.dataset(cfg.data_path, 'test', transform=datasets.augmentation_valid())
    
    train_loader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(validset, batch_size=cfg.val_batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=cfg.val_batch_size, shuffle=False)
    # -------------------------------------------------------------------
    # load loss
    lossfn = load_lossfn(cfg.lossfn)
    # -------------------------------------------------------------------
    # load network
    net = load_network(cfg)
    net = net.to(device)
    # -------------------------------------------------------------------
    # load optimizer
    optimizer = load_optimizer(net, cfg)
#      opt = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
#         sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs/2)
    # -------------------------------------------------------------------
    # start train
        
#         torchvision.utils.save_image(torchvision.utils.make_grid(torch.cat((haze_image, dehaze_image, ori_image), 0),nrow = ori_image.shape[0]),os.path.join(cfg.sample_output_folder, 'w{}_{}.jpg'.format(epoch , step)))
    

#     optimizer = optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=max(-1, 0))
#     scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=cfg.epochs)
    ###### train
    net = net.to(device)
    
    if cfg.experiment_name == None:
        cfg.experiment_name = "Net{}_Loss{}_Norm{}_".format(cfg.net, cfg.lossfn,cfg.net_norm)
    print('Current Experiment:',cfg.experiment_name)

    wandb.init(name=cfg.experiment_name)
    wandb.run.name = cfg.experiment_name + wandb.run.id
    wandb.config.update(cfg) # adds all of the arguments as config variables
    wandb.watch(net, lossfn, log="all", log_freq=10)

    def train(train_loader,status='train'):
        net.train()
        loss_temp = list()
        metric_temp = list()
        for idx,batch in enumerate(train_loader):
            x,y = batch['x'].to(device),batch['y'].to(device)
            yhat = net(x)
            yhat = softmax_T(yhat)
            loss = lossfn(yhat,y)
            loss_temp.append(loss.cpu().detach().numpy())
            dice = f1_score(y.cpu().detach().numpy().flatten(),torch.argmax(yhat,1).cpu().detach().numpy().flatten())       
            metric_temp.append(dice)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip_norm)
            optimizer.step()
        wandb.log({status : wb_mask(x, yhat, y)})
        
        loss_train.append(np.mean(np.array(loss_temp)))
        metric_train.append(np.mean(np.array(metric_temp)))

    def test(test_loader,status='valid'):
        net.eval()
        loss_temp = list()
        metric_temp = list()
        
        for idx,batch in enumerate(test_loader):
            with torch.no_grad():
                x,y = batch['x'].to(device),batch['y'].to(device)
                yhat = net(x)
                yhat = softmax_T(yhat)
                loss = lossfn(yhat,y)
                loss_temp.append(loss.cpu().detach().numpy())
                dice = f1_score(y.cpu().detach().numpy().flatten(),torch.argmax(yhat,1).cpu().detach().numpy().flatten())   
                metric_temp.append(dice)
        wandb.log({status: wb_mask(x, yhat, y)})
                
        if status =='valid':
            loss_valid.append(np.mean(np.array(loss_temp)))
            metric_valid.append(np.mean(np.array(metric_temp)))
        elif status =='test':
            loss_test.append(np.mean(np.array(loss_temp)))
            metric_test.append(np.mean(np.array(metric_temp)))

            
    loss_train = list()
    loss_valid = list()
    loss_test = list()
    metric_train = list()
    metric_valid = list()    
    metric_test = list()

    for epoch in range(cfg.epochs):

        train(train_loader,'train')
        test(valid_loader,'valid')
        test(test_loader,'test')

        if epoch>5 and np.max(np.array(metric_test)) == metric_test[-1]:
            save_model(net,cfg)
            
        wandb.log({"epoch": epoch,
                   "loss_train":loss_train[-1],
                   "loss_valid":loss_valid[-1],
                   "loss_test":loss_test[-1],
                   "metric_train": metric_train[-1],
                   "metric_valid":metric_valid[-1],
                   "metric_test":metric_test[-1],
                   "metric_test_max":max(metric_test),
                   "lr":optimizer.param_groups[0]['lr']
                  })
        
if __name__ == '__main__':
    config_args, unparsed_args = get_config()
    main(config_args)
