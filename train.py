import warnings
warnings.filterwarnings(action='ignore')

from argparse import ArgumentParser, Namespace

import os, random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

import monai
import nets
import losses
import datasets
import utils
import wandb
import torchmetrics

# import kornia
# import cv2
# import matplotlib.pyplot as plt

from monai.inferers import sliding_window_inference

class SegModel(pl.LightningModule):

    def __init__(
        self,
        data_dir: str,
        data_module = 'dataset',
        batch_size: int = 1,
        finetune = True,
        gpus = -1,
        net_inputch = 1,
        net_outputch = 1,
        lossfn = 'CELoss',
        net = 'segunet',   
        precision = 32,
        data_padsize= None,
        data_cropsize= None,
        data_resize= None,
        data_patchsize= None,
        experiment_name = None,
        **kwargs):
                
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.data_module = data_module
        self.batch_size = batch_size
        self.finetune = finetune
        self.net_inputch = net_inputch
        self.net_outputch = net_outputch
        self.precision = precision
        self.data_padsize = data_padsize
        self.data_cropsize = data_cropsize
        self.data_resize = data_resize
        self.data_patchsize = data_patchsize
        
        # loss       
        fn_call = getattr(losses, lossfn)
        self.lossfn = fn_call()

        # net
        fn_call = getattr(nets, net)
        self.net = fn_call(net_inputch=self.net_inputch, net_outputch=self.net_outputch)
        
        # metric
        self.metric = torchmetrics.F1(self.net_outputch)
        
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x,y  = batch['x'], batch['y']

        yhat = self(x)
        yhat = utils.Activation(yhat)
        loss = self.lossfn(yhat, y)
        metric = self.metric(torch.argmax(yhat,1).cpu().int().flatten(),y.cpu().int().flatten())

        self.log('loss', loss, on_step=True, prog_bar=True)
        self.log('metric', metric, on_step=True, prog_bar=True)
        self.logger.experiment.log({'image_train' : wb_mask(x, yhat, y)}) # wandb.log({'train' : wb_mask(x, yhat, y)})
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        x,y  = batch['x'], batch['y']
        
#         yhat = self(x)
        roi_size = self.data_patchsize if isinstance(self.data_patchsize, int) else int(self.data_patchsize.split('_')[0]) 
        yhat = sliding_window_inference(inputs=x,roi_size=roi_size,sw_batch_size=4,predictor=self.net,overlap=0.5,mode='constant')
        yhat = utils.Activation(yhat)
        loss = self.lossfn(yhat, y)
        metric = self.metric(torch.argmax(yhat,1).cpu().int().flatten(),y.cpu().int().flatten())

        self.log('loss_val', loss, on_step=True, prog_bar=True)
        self.log('metric_val', metric, on_step=True, prog_bar=True)
        self.logger.experiment.log({'image_val' : wb_mask(x, yhat, y)})
        return {'loss_val': loss}    

    def configure_optimizers(self):
        """
        mode : finetune --> Adam with lr = 1e-4
        mode : CosineAnnealingWarmup (default) --> SGD with varying lr
        """
        if self.finetune == 'True':
            optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)    
        else:
            optimizer = torch.optim.SGD(self.net.parameters(), lr=0)
            scheduler = utils.CosineAnnealingWarmUpRestarts(optimizer, T_0=200, T_mult=1, eta_max=0.01, T_up=10, gamma=0.5)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler,
                                 'monitor': 'loss_val'}
                }
    
    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument("--data_dir", type=str, help="path where dataset is stored, subfolders name should be x_train, y_train")
        parser.add_argument("--data_module", type=str,default='dataset', help="Data Module, see datasets.py")
        parser.add_argument("--data_padsize", type=str, default=None, help="input like this (height_width) : pad - crop - resize - patch")
        parser.add_argument("--data_cropsize", type=str, default=None, help="input like this (height_width) : pad - crop - resize - patch")
        parser.add_argument("--data_resize", type=str, default=None, help="input like this (height_width) : pad - crop - resize - patch")
        parser.add_argument("--data_patchsize", type=str, default=None, help="input like this (height_width) : pad - crop - resize - patch: recommand (A * 2^n)")
        parser.add_argument("--lossfn", type=str, default='CE', help="[CELoss, DiceCELoss, MSE, ...], see losses.py")
        parser.add_argument("--net", type=str, default='segunet', help="Networks, see nets.py")
        parser.add_argument("--net_inputch", type=int, default=1, help='dimension of input channel')
        parser.add_argument("--net_outputch", type=int, default=2, help='dimension of output channel')        
        parser.add_argument("--precision", type=int, default=32, help='amp will be set when 16 is given')
        parser.add_argument("--finetune", type=str, default='False', help="Set Adam with lr=1e-4")        
        parser.add_argument("--experiment_name", type=str, default=None, help='Postfix name of experiment')         
        return parser

class MyDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = "path/to/dir", data_module ='dataset', classes='all', batch_size: int = 32, data_padsize = None, data_cropsize= None, data_resize= None, data_patchsize = None, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.data_module = data_module
        self.classes = classes
        self.batch_size = batch_size 
        self.data_padsize = data_padsize
        self.data_cropsize = data_cropsize
        self.data_resize = data_resize
        self.data_patchsize = data_patchsize
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        
        fn_call = getattr(datasets, self.data_module)
#         classes = 'only_background'
        classes = 'only_vessel'
        self.trainset = fn_call(self.data_dir, 'train', classes = classes, 
                                transform_spatial =datasets.augmentation_image_size(data_padsize = self.data_padsize,
                                                                                 data_cropsize = self.data_cropsize,
                                                                                 data_resize = self.data_resize,
                                                                                 data_patchsize = self.data_patchsize,), 
                                transform=datasets.augmentation_train())
        self.validset = fn_call(self.data_dir, 'valid', classes = classes, transform=datasets.augmentation_valid())
        self.testset = fn_call(self.data_dir, 'test', classes = classes, transform=datasets.augmentation_valid())
        
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


# wandb image
segmentation_classes = ['black', 'class1', 'class2']

def labels():
    l = {}
    for i, label in enumerate(segmentation_classes):
        l[i] = label
    return l

def wb_mask(x, yhat, y, samples=2):
    x = torchvision.utils.make_grid(x[:samples].cpu().detach(),normalize=True).permute(1,2,0)
    y = torchvision.utils.make_grid(y[:samples].cpu().detach()).permute(1,2,0)
    yhat = torchvision.utils.make_grid(torch.argmax(yhat[:samples],1).unsqueeze(1).cpu()).permute(1,2,0) if yhat.shape[1]>1 else \
           torchvision.utils.make_grid(yhat[:samples].round().cpu()).permute(1,2,0)
        
    x = (x*255).numpy().astype(np.uint8)        # 0 ~ 255
    yhat = yhat[...,0].numpy().astype(np.uint8) # 0 ~ n_class 
    y = y[...,0].numpy().astype(np.uint8)       # 0 ~ n_class
    
    return wandb.Image(x, masks={
    "prediction" : {"mask_data" : yhat, "class_labels" : labels()},
    "ground truth" : {"mask_data" : y, "class_labels" : labels()}})


def main(args: Namespace):
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = SegModel(**vars(args))
    
    # ------------------------
    # 2 SET LOGGER
    # ------------------------    
    from pytorch_lightning import loggers as pl_loggers
    from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor, StochasticWeightAveraging, LambdaCallback, EarlyStopping
    
    args.experiment_name = "Net{}_Netinputch{}_Netoutputch{}_Loss{}_Precision{}_Patchsize{}_Prefix{}_"\
    .format(args.net, args.net_inputch, args.net_outputch, args.lossfn, args.precision,args.data_patchsize,args.experiment_name)
    print('Current Experiment:',args.experiment_name)
    
    wb_logger = pl_loggers.WandbLogger(save_dir='logs/', name=args.experiment_name)
    wb_logger.log_hyperparams(args)
    
    wandb.init(name=args.experiment_name)
    wandb.run.name = args.experiment_name + wandb.run.id
    wandb.config.update(args, allow_val_change=True) 
    
    checkpoint_callback = ModelCheckpoint(verbose=True, 
#                                           monitor='loss_val',
#                                           mode='min',
                                          monitor='metric_val',
                                          mode='max',
                                          filename='{epoch:04d}-{loss_val:.4f}-{metric_val:.4f}',
                                          save_top_k=3,)
    
    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer.from_argparse_args(args,
#                                             accelerator=accelerator,
                                            amp_backend='native',
                                            gpus = -1,
                                            sync_batchnorm =True,
                                            callbacks=[checkpoint_callback,
                                                       LearningRateMonitor(),
                                                       StochasticWeightAveraging(),
#                                                        EarlyStopping(monitor='loss_val',patience=200),
                                                       EarlyStopping(monitor='metric_val',patience=200),
                                                      ],
                                            auto_scale_batch_size='power',
                                            weights_summary='top', 
                                            log_gpu_memory='min_max',
                                            max_epochs=2000,
                                            deterministic=True,
                                            num_processes=4,
                                            stochastic_weight_avg=True,
                                            logger=wb_logger
                                           )
    
    myData = MyDataModule.from_argparse_args(args) 
    trainer.tune(model,datamodule=myData)
    trainer.fit(model,datamodule=myData)
    
    # ------------------------
    # 5 START TRAINING
    # ------------------------
if __name__ == '__main__':
    
    parser = ArgumentParser(add_help=False)
    parser = SegModel.add_model_specific_args(parser)
    
    args = parser.parse_args()
    print('args:',args,'\n')
    
    main(args)
    