import warnings
warnings.filterwarnings("ignore")

import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True
})
from utils import * 
import utils

from config import get_config
import config

# @logger
def save_model(net, optimizer, cfg):
    if torch.cuda.device_count() == 1:
#         torch.save(net.state_dict(),cfg.experiment_name+'.pt')
        state_dict = {'net_state_dict':net.state_dict(),
                      'optimizer_state_dict':optimizer.state_dict()}
        torch.save(state_dict,cfg.experiment_name+'.pt')
    else:
#         torch.save(net.module.state_dict(),cfg.experiment_name+'.pt')
        state_dict = {'net_state_dict':net.module.state_dict(),
                      'optimizer_state_dict':optimizer.state_dict()}
        torch.save(state_dict,cfg.experiment_name+'.pt')

    print('weight saved')

# @logger
def load_optimizer(net, cfg):
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return optimizer

# @logger
def load_network(cfg):
    import segmentation_models_pytorch as smp
    if cfg.net =='manet':
        import segmentation_models_pytorch as smp
        net = smp.MAnet('timm-efficientnet-b7', in_channels=cfg.net_inputch, classes=cfg.net_outputch)
    elif cfg.net =='waveletunet':
        net= models.Waveletunet(in_channel=cfg.net_inputch,out_channel=cfg.net_outputch, c=32, norm=cfg.net_norm,mc_dropout=cfg.net_mcdropout)
        net.apply(weight_init)
    elif cfg.net =='attunet':
        net = models.AttU_Net(img_ch=cfg.net_inputch,output_ch=cfg.net_outputch,norm=cfg.net_norm,mc_dropout=cfg.net_mcdropout)
    elif cfg.net =='r2attunet':
        net = models.R2AttU_Net(img_ch=cfg.net_inputch,output_ch=cfg.net_outputch,t=2,norm=cfg.net_norm,mc_dropout=cfg.net_mcdropout,nnblock=cfg.net_nnblock)
    elif cfg.net == 'tightwaveletnet':
        net = models.TightWaveletnet(in_channel=cfg.net_inputch, out_channel=cfg.net_outputch, c=32, norm=cfg.net_norm)
    elif cfg.net == 'manet':
        net = smp.MAnet(
                        encoder_name='timm-efficientnet-b5',
                        encoder_depth=5,
                        encoder_weights='imagenet',
                        decoder_use_batchnorm=True,
                        decoder_channels=(256, 128, 64, 32, 16),
                        decoder_pab_channels=64,
                        in_channels=cfg.net_inputch,
                        classes=cfg.net_outputch,
                        activation=None,
                        aux_params=None)
    elif cfg.net == 'axialattnet':
        net = models.axial50l(in_channel=1,)
    else:
        print('default net')      
        net = smp.MAnet(
                        encoder_name='timm-efficientnet-b5',
                        encoder_depth=5,
                        encoder_weights='imagenet',
                        decoder_use_batchnorm=True,
                        decoder_channels=(256, 128, 64, 32, 16),
                        decoder_pab_channels=64,
                        in_channels=cfg.net_inputch,
                        classes=cfg.net_outputch,
                        activation=None,
                        aux_params=None)

        
    if cfg.net_pretrained is not None:
        try:
            weight = torch.load(cfg.net_pretrained)
            net.load_state_dict(weight,strict=True)
            print("loading weight succeed")
            del weight
        except:
            print('loading weight failed, check once again')
        
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)        
    return net

# @logger
def load_lossfn(cfg):
    if cfg.lossfn =='dicece':
        lossfn = losses.DiceCELoss()
    elif cfg.lossfn =='boundaryce':
        lossfn = losses.BoundaryCELoss()
    elif cfg.lossfn =='cldicece':
        lossfn = losses.clDiceCELoss()
    elif cfg.lossfn =='bce':
        lossfn = nn.BCEWithLogitsLoss()
    elif cfg.lossfn =='ce':
        lossfn = losses.CrossEntropyLoss()
    else:
        print('default loss')
        lossfn = losses.CrossEntropyLoss()
    return lossfn

# wandb image
segmentation_classes = ['background', 'vessel']

def labels():
    l = {}
    for i, label in enumerate(segmentation_classes):
        l[i] = label
    return l

def wb_mask(x, yhat, y):
    x = x.cpu().detach().numpy()[0]
    y = y.cpu().detach().numpy()[0]
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

def main(cfg):
    # -------------------------------------------------------------------
    print(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = torch.cuda.amp.GradScaler()
    
    if cfg.half_precision==True:
        print('\nYou are using {} gpu(s) + using half precision'.format(torch.cuda.device_count()))
    else:
        print('\nYou are using {} gpu(s)'.format(torch.cuda.device_count()))
    
    # load data
    trainset = datasets.dataset(cfg.data_path, 'train', transform=datasets.augmentation_train())
    validset = datasets.dataset(cfg.data_path, 'valid', transform=datasets.augmentation_valid())
    testset = datasets.dataset(cfg.data_path, 'test', transform=datasets.augmentation_valid())
    train_loader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(validset, batch_size=cfg.val_batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=cfg.val_batch_size, shuffle=False, pin_memory=True)
    
    # check dataset
    batch = next(iter(train_loader))
    x,y = batch['x'], batch['y']
    print('x',x.shape)
    print('y',y.shape,'\n')    
    # -------------------------------------------------------------------

    # load loss
    lossfn = load_lossfn(cfg)
    # -------------------------------------------------------------------
    # load network
    net = load_network(cfg)        
    net = net.to(device)
    # -------------------------------------------------------------------
    # load optimizer
    optimizer = load_optimizer(net, cfg)

    cfg.experiment_name = "Net{}_Loss{}_Norm{}_Prefix{}_".format(cfg.net, cfg.lossfn, cfg.net_norm, cfg.experiment_name)
    print('Current Experiment:',cfg.experiment_name)

    wandb.init(name=cfg.experiment_name)
    wandb.run.name = cfg.experiment_name + wandb.run.id
    wandb.config.update(cfg) # adds all of the arguments as config variables
    wandb.watch(net, lossfn, log="all", log_freq=100)

    def Activation(tensor,Temperature=1):
        if tensor.shape[1]>1:
            tensor = F.softmax(tensor/Temperature,1)
        else:
            tensor = 1 / (1 + torch.exp(-tensor/Temperature))
#             tensor = top_hat(tensor,torch.ones(11,11).to(device))
#             tensor = kornia.enhance.normalize_min_max(tensor)
        return tensor
    
    def train(train_loader, status='train'):
        net.train()
        loss_temp = list()
        metric_temp = list()
        for idx,batch in enumerate(train_loader):
#             x,y = batch['x'].to(device,non_blocking=True),batch['y'].to(device,non_blocking=True)
            x,y = batch['x'].to(device),batch['y'].to(device)
            if cfg.half_precision == False:
                optimizer.zero_grad()
                yhat = net(x)
                yhat = Activation(yhat)
                loss = lossfn(yhat,y)
                loss.backward()
#                 torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip_norm)
                optimizer.step()
            else:
                with torch.cuda.amp.autocast():
                    optimizer.zero_grad()
                    yhat = net(x)
                    yhat = Activation(yhat)
                    loss = lossfn(yhat,y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            loss_temp.append(loss.cpu().detach().numpy())

            if y.shape != yhat.shape:
                dice = f1_score(y.cpu().detach().numpy().flatten(),torch.argmax(yhat,1).cpu().detach().numpy().flatten())       
            else:
                dice = f1_score(y.cpu().detach().numpy().flatten(),yhat.round().cpu().detach().numpy().flatten())       
            metric_temp.append(dice)
        wandb.log({status : wb_mask(x, yhat, y)})
        
        loss_train.append(np.mean(np.array(loss_temp)))
        metric_train.append(np.mean(np.array(metric_temp)))

    def test(test_loader, status='valid'):
        net.eval()
        loss_temp = list()
        metric_temp = list()
        
        for idx,batch in enumerate(test_loader):
            with torch.no_grad():
                x,y = batch['x'].to(device,non_blocking=True),batch['y'].to(device,non_blocking=True)
                if cfg.half_precision == False:
                    yhat = net(x)
                    yhat = Activation(yhat)
                    loss = lossfn(yhat,y)
                else:
                    with torch.cuda.amp.autocast():
                        yhat = net(x)
                        yhat = Activation(yhat)
                        loss = lossfn(yhat,y)
                        scaler.scale(loss)
                loss_temp.append(loss.cpu().detach().numpy())
                if y.shape != yhat.shape:
                    dice = f1_score(y.cpu().detach().numpy().flatten(),torch.argmax(yhat,1).cpu().detach().numpy().flatten())       
                else:
                    dice = f1_score(y.cpu().detach().numpy().flatten(),yhat.round().cpu().detach().numpy().flatten())       
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
        print('\nepoch:{}, loss_train:{},loss_valid:{},loss_test:{}'.format(epoch,loss_train[-1],loss_valid[-1],loss_test[-1]))
        print('epoch:{}, metric_train:{},metric_valid:{},metric_test:{}'.format(epoch,metric_train[-1],metric_valid[-1],metric_test[-1]))

        if epoch>5 and np.max(np.array(metric_test)) == metric_test[-1]:
            save_model(net, optimizer, cfg)
            
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