# Mytorch for segmentation
- pytorch-lighting based segmentation model
- support sweeps using wandb

# HOW TO USE
```python 
   usage: train.py [-h] [--data_dir DATA_DIR] [--data_module DATA_MODULE]
                [--finetune FINETUNE] [--data_padsize DATA_PADSIZE]
                [--data_cropsize DATA_CROPSIZE] [--data_resize DATA_RESIZE]
                [--data_patchsize DATA_PATCHSIZE] [--lossfn LOSSFN]
                [--net NET] [--net_inputch NET_INPUTCH]
                [--net_outputch NET_OUTPUTCH] [--precision PRECISION]
                [--experiment_name EXPERIMENT_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   path where dataset is stored, subfolders name should
                        be x_train, y_train
  --data_module DATA_MODULE
                        Data Module, see datasets.py
  --finetune FINETUNE   Set Adam with lr=1e-4
  --data_padsize DATA_PADSIZE
                        input like this (height_width) : pad - crop - resize - patch
  --data_cropsize DATA_CROPSIZE
                        input like this (height_width) : pad - crop - resize - patch
  --data_resize DATA_RESIZE
                        input like this (height_width) : pad - crop - resize - patch
  --data_patchsize DATA_PATCHSIZE
                        input like this (height_width) : pad - crop - resize - patch: recommand (A * 2^n)
  --lossfn LOSSFN       [CELoss, DiceCELoss, MSE, ...], see losses.py
  --net NET             Networks, see nets.py
  --net_inputch NET_INPUTCH
                        dimension of input channel
  --net_outputch NET_OUTPUTCH
                        dimension of output channel
  --precision PRECISION
                        amp will be set when 16 is given
  --experiment_name EXPERIMENT_NAME
                        Postfix name of experiment
```
