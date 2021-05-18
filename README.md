# Mytorch
# HOW TO USE
usage: train_seg.py [-h] [--data_path DATA_PATH]
                    [--weightsave_path WEIGHTSAVE_PATH] [--lr LR]
                    [--lr_warmup LR_WARMUP] [--num_workers NUM_WORKERS]
                    [--batch_size BATCH_SIZE]
                    [--val_batch_size VAL_BATCH_SIZE] [--epochs EPOCHS]
                    [--weight_decay WEIGHT_DECAY]
                    [--grad_clip_norm GRAD_CLIP_NORM]
                    [--experiment_name EXPERIMENT_NAME] [--lossfn LOSSFN]
                    [--net NET] [--net_inputch NET_INPUTCH]
                    [--net_outputch NET_OUTPUTCH]
                    [--net_pretrained NET_PRETRAINED]
                    [--net_mcdropout NET_MCDROPOUT] [--net_norm NET_NORM]
                    [--net_nnblock NET_NNBLOCK]
                    [--half_precision HALF_PRECISION]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Origin image path
  --weightsave_path WEIGHTSAVE_PATH
                        Origin image path
  --lr LR               Learning Rate. Default=1e-4
  --lr_warmup LR_WARMUP
                        Learning Rate warmup scheduler
  --num_workers NUM_WORKERS
                        Number of threads for data loader, for window set to 0
  --batch_size BATCH_SIZE
                        Training batch size
  --val_batch_size VAL_BATCH_SIZE
                        Validation batch size
  --epochs EPOCHS       number of epochs for training
  --weight_decay WEIGHT_DECAY
  --grad_clip_norm GRAD_CLIP_NORM
  --experiment_name EXPERIMENT_NAME
                        name of experiment
  --lossfn LOSSFN       [ce,dicece,boundaryce,cldicece]
  --net NET             Networks, see models.py
  --net_inputch NET_INPUTCH
                        dimension of input channel
  --net_outputch NET_OUTPUTCH
                        dimension of output channel
  --net_pretrained NET_PRETRAINED
                        path to weights,
  --net_mcdropout NET_MCDROPOUT
                        MC dropout rate of bottleneck,
  --net_norm NET_NORM   batch,instance,group,ws
  --net_nnblock NET_NNBLOCK
                        True,False
  --half_precision HALF_PRECISION
                        True,False
