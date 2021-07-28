import warnings
warnings.filterwarnings(action='ignore')

from argparse import ArgumentParser, Namespace

import os, random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import segmentation_models_pytorch as smp
import pytorch_lightning as pl

import monai
import nets
import losses
import datasets
import utils
import wandb

import kornia
import cv2
import matplotlib.pyplot as plt

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
        patch_size= 128,
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
        self.patch_size = patch_size
        
        # loss       
        fn_call = getattr(losses, lossfn)
        self.lossfn = fn_call()
        
        fn_call = getattr(nets, net)
        self.net = fn_call(net_inputch=self.net_inputch, net_outputch=self.net_outputch)
        
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x,y  = batch['x'], batch['y']

        yhat = self(x)
        yhat = utils.Activation(yhat)
        loss = self.lossfn(yhat, y)
#         metric = self.metrics(torch.argmax(yhat,1),y,'')
#         metric['loss'] = np.asscalar(loss.cpu().detach().numpy())

        self.log('loss', loss, prog_bar=True)
#         self.log('accuracy',metric['accuracy'],on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         self.log('dice',metric['dice'],on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         self.log('sensitivity',metric['sensitivity'],on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         self.log('specificity',metric['specificity'],on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
#         self.logger.experiment.add_image('train_x', x[0],self.current_epoch)
#         self.logger.experiment.add_image('train_y', y[0],self.current_epoch)
#         self.logger.experiment.add_image('train_yhat', torch.argmax(yhat,1).unsqueeze(1)[0],self.current_epoch)

#         self.logger.log_metrics(metric)
#         self.logger.save()
        
#         return {'loss': loss, 'x': x, 'yhat': yhat}
#         save_samples(x,y,yhat,'train_sample')
        wandb.log({'train' : wb_mask(x, yhat, y)})
#         wandb.log({status : wb_mask(x, yhat, y)})
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        x,y  = batch['x'], batch['y']
        
#         yhat = self(x)
        roi_size = self.patch_size
        yhat = sliding_window_inference(inputs=x,roi_size=roi_size,sw_batch_size=4,predictor=self.net,overlap=0.5,mode='constant')
        yhat = utils.Activation(yhat)
        loss = self.lossfn(yhat, y)
#         metric = self.metrics(torch.argmax(yhat,1),y,'')        
#         metric['loss_val'] = np.asscalar(loss.cpu().detach().numpy())
        
        self.log('loss_val', loss, prog_bar=True)
#         self.log('accuracy_val',metric['accuracy'],on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         self.log('dice_val',metric['dice'],on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         self.log('sensitivity_val',metric['sensitivity'],on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         self.log('specificity_val',metric['specificity'],on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
#         self.logger.experiment.add_image('valid_x', x[0],self.current_epoch)
#         self.logger.experiment.add_image('valid_y', y[0],self.current_epoch)
#         self.logger.experiment.add_image('valid_yhat', torch.argmax(yhat,1).unsqueeze(1)[0],self.current_epoch)
#         self.logger.experiment.log({
#                     "val/examples": wandb.Image(x[0], caption='aa')})
#         self.logger.experiment.log('valid',x[0], self.current_epoch)
        self.logger.experiment.log({'valid' : wb_mask(x, yhat, y)})
    
#         self.logger.log_metrics(metric)
#         self.logger.save()
#         wandb.log({'valid' : wb_mask(x, yhat, y)})
#         save_samples(x,y,yhat,'valid_sample')
        
        return {'loss_val': loss}    

    def configure_optimizers(self):

        if self.finetune == 'True':
            optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)    
        else:
            optimizer = torch.optim.SGD(self.net.parameters(), lr=0)
            scheduler = utils.CosineAnnealingWarmUpRestarts(optimizer, T_0=100, T_mult=1, eta_max=0.1, T_up=10, gamma=0.5)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler,
                                 'monitor': 'loss_val'}
                }
               
    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument("--data_dir", type=str, help="path where dataset is stored, subfolders name should be x_train, y_train")
        parser.add_argument("--data_module", type=str,default='dataset', help="Data Module")
        parser.add_argument("--finetune", type=str, default='False', help="Set Adam with lr=1e-4")        
        parser.add_argument("--patch_size", type=int, default=128, help="data patch_size")
        parser.add_argument("--lossfn", type=str, default='CE', help="[CELoss, DiceCELoss, MSE], see losses.py")
        parser.add_argument("--net", type=str, default='segunet', help="Networks, see nets.py")
        parser.add_argument("--net_inputch", type=int, default=1, help='dimension of input channel')
        parser.add_argument("--net_outputch", type=int, default=2, help='dimension of output channel')        
        parser.add_argument("--precision", type=int, default=32, help='amp will be set when 16 is given')
        parser.add_argument("--experiment_name", type=str, default=None, help='name of experiment')         
        return parser

class MyDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = "path/to/dir", data_module ='dataset', classes='all', batch_size: int = 32, patch_size = 128, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.data_module = data_module
        self.classes = classes
        self.batch_size = batch_size 
        self.patch_size = patch_size
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        
        fn_call = getattr(datasets, self.data_module)
#         classes = 'only_background'
        classes = 'only_vessel'
        self.trainset = fn_call(self.data_dir, 'train', classes = classes, transform_crop =datasets.augmentation_crop(patch_size = self.patch_size), transform=datasets.augmentation_train())
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

def wb_mask(x, yhat, y):
    x = x.cpu().detach().numpy()[0]
    y = y.cpu().detach().numpy()[0]
#     x = kornia.tensor_to_numpy()[0]
#     y = kornia.tensor_to_numpy()[0]
    if yhat.shape[1]>1:
        yhat = torch.argmax(yhat,1).cpu().detach().numpy()[0]
    else:
        yhat = yhat.cpu().detach().numpy().round()[0,0]
        yhat = yhat.astype(np.uint8)
    x = np.moveaxis(x,0,-1)
    y = y[0]
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
    .format(args.net, args.net_inputch, args.net_outputch, args.lossfn, args.precision,args.patch_size,args.experiment_name)
    print('Current Experiment:',args.experiment_name)
    
    wb_logger = pl_loggers.WandbLogger(save_dir='logs/', name=args.experiment_name)
    wb_logger.log_hyperparams(args)
    
    wandb.init(name=args.experiment_name)
    wandb.run.name = args.experiment_name + wandb.run.id
    wandb.config.update(args) # adds all of the arguments as config variables
    
    checkpoint_callback = ModelCheckpoint(verbose=True, 
                                          monitor='loss_val',
                                          mode='min',
                                          filename='{epoch:04d}-{loss_val:.4f}',
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
                                                       EarlyStopping(monitor='loss_val',patience=150)],
                                            auto_scale_batch_size='power',
                                            weights_summary='top', 
                                            log_gpu_memory='min_max',
                                            max_epochs=1000,
                                            deterministic=True,
                                            num_processes=4,
                                            stochastic_weight_avg=True,
                                            logger=wb_logger
                                           )
    # ------------------------
    # 5 START TRAINING
    # ------------------------
    myData = MyDataModule.from_argparse_args(args) 
    trainer.tune(model,datamodule=myData)
    trainer.fit(model,datamodule=myData)

if __name__ == '__main__':
    
    parser = ArgumentParser(add_help=False)
    parser = SegModel.add_model_specific_args(parser)
    
    args = parser.parse_args()
    print('args:',args,'\n')
    
    main(args)
    