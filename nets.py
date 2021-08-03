import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import monai
import segmentation_models_pytorch as smp

groupnorm_parameter = 4

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def bn2instance(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.InstanceNorm2d(module.num_features,
                                                module.eps, module.momentum,
                                                module.affine,
                                                module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig

    for name, child in module.named_children():
        module_output.add_module(name, bn2instance(child))

    del module
    return module_output

def bn2group(module):
    num_groups = 4 # hyper_parameter of GroupNorm
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.GroupNorm(num_groups,
                                           module.num_features,
                                           module.eps, 
                                           module.affine,
                                          )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig

    for name, child in module.named_children():
        module_output.add_module(name, bn2group(child))

    del module
    return module_output

class manet_eb5(nn.Module):
    def __init__(self, net_inputch=3, net_outputch=2):
        super(manet_eb5, self).__init__()        
        self.net_inputch = net_inputch
        self.net_outputch = net_outputch  
        self.net = smp.MAnet(
                        encoder_name='timm-efficientnet-b5',
                        encoder_depth=5,
                        encoder_weights='imagenet',
                        decoder_use_batchnorm=True,
                        decoder_channels=(256, 128, 64, 32, 16),
                        decoder_pab_channels=64,
                        in_channels=self.net_inputch, 
                        classes=self.net_outputch,
                        activation=None,
                        aux_params=None)
        try:
            weight = torch.load('NetNone_Lossdicece_Normws_Prefixcase4_.pt')
            self.net.load_state_dict(weight['net_state_dict'])
        except:
            pass
        
    def forward(self,x):
        return self.net(x)    
    
class unet_eb5(nn.Module):
    def __init__(self, net_inputch=3, net_outputch=2):
        super(unet_eb5, self).__init__()        
        self.net_inputch = net_inputch
        self.net_outputch = net_outputch  
        self.net = smp.Unet(
                        encoder_name='timm-efficientnet-b5',
                        decoder_use_batchnorm = True,
                        decoder_attention_type = 'scse',
                        encoder_depth=5,
                        in_channels=self.net_inputch, 
                        classes=self.net_outputch,)
    def forward(self,x):
        return self.net(x)    

class manet_elb4_batch(nn.Module):
    def __init__(self, net_inputch=3, net_outputch=2):
        super(manet_elb4_batch, self).__init__()        
        self.net_inputch = net_inputch
        self.net_outputch = net_outputch  
        self.net = smp.MAnet(
                        encoder_name='timm-tf_efficientnet_lite4',
                        encoder_depth=5,
                        encoder_weights='imagenet',
                        decoder_use_batchnorm=True,
                        decoder_channels=(256, 128, 64, 32, 16),
                        decoder_pab_channels=64,
                        in_channels=self.net_inputch, 
                        classes=self.net_outputch,
                        activation=None,
                        aux_params=None)
    
    def forward(self,x):
        return self.net(x)    
    
class segresnet(nn.Module):
    def __init__(self,net_inputch=3,net_outputch=3):
        super(segresnet, self).__init__()        
        self.net_inputch = net_inputch
        self.net_outputch = net_outputch
        self.net = monai.networks.nets.SegResNet(2,8,in_channels=self.net_inputch, out_channels=self.net_outputch)
        self.net = bn2instance(self.net)

    def forward(self,x):
        return self.net(x)
    
# wavelet Unet
import pywt
import torch
from torch.autograd import Variable

w=pywt.Wavelet('db1')
dec_hi = torch.Tensor(w.dec_hi[::-1]) 
dec_lo = torch.Tensor(w.dec_lo[::-1])
rec_hi = torch.Tensor(w.rec_hi)
rec_lo = torch.Tensor(w.rec_lo)
filters = torch.stack([dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1)/2.0,
                       dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1),
                       dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1),
                       dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)], dim=0)
inv_filters = torch.stack([rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1)*2.0,
                           rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1),
                           rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1),
                           rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)], dim=0)

def wt(vimg):
    padded = vimg
    res = torch.zeros(vimg.shape[0],4*vimg.shape[1],int(vimg.shape[2]/2),int(vimg.shape[3]/2))
    res = res.cuda()
    for i in range(padded.shape[1]):
        res[:,4*i:4*i+4] = torch.nn.functional.conv2d(padded[:,i:i+1], Variable(filters[:,None].cuda(),requires_grad=True),stride=2)
        res[:,4*i+1:4*i+4] = (res[:,4*i+1:4*i+4]+1)/2.0
    return res

def iwt(vres):
    res = torch.zeros(vres.shape[0],int(vres.shape[1]/4),int(vres.shape[2]*2),int(vres.shape[3]*2))
    res = res.cuda()
    for i in range(res.shape[1]):
        vres[:,4*i+1:4*i+4]=2*vres[:,4*i+1:4*i+4]-1
        temp = torch.nn.functional.conv_transpose2d(vres[:,4*i:4*i+4], Variable(inv_filters[:,None].cuda(),requires_grad=True),stride=2)
        res[:,i:i+1,:,:] = temp
    return res

# class waveletunet_batch(nn.Module):
#     def __init__(self,net_inputch=3,net_outputch=2,num_c=4):
#         super(waveletunet_batch,self).__init__()        
        
#         self.Conv1 = conv_block(ch_in=net_inputch,ch_out=num_c)
#         self.Conv2 = conv_block(ch_in=num_c*4,ch_out=num_c*4)
#         self.Conv3 = conv_block(ch_in=num_c*16,ch_out=num_c*16)
#         self.Conv4 = conv_block(ch_in=num_c*64,ch_out=num_c*64)
#         self.Conv5 = conv_block(ch_in=num_c*256,ch_out=num_c*256)

#         self.Up_conv5 = conv_block(ch_in=num_c*128, ch_out=num_c*64)
#         self.Up_conv4 = conv_block(ch_in=num_c*32, ch_out=num_c*16)
#         self.Up_conv3 = conv_block(ch_in=num_c*8, ch_out=num_c*4)
#         self.Up_conv2 = conv_block(ch_in=num_c*2,ch_out=num_c)

#         self.Conv_1x1 = nn.Conv2d(num_c,net_outputch,kernel_size=1,stride=1,padding=0)

#     def forward(self,x):
# #         print('x',x.shape)

#         # encoding path
#         x1 = self.Conv1(x)
# #         print('x1',x1.shape)

#         x2 = wt(x1)
# #         x2 = self.Maxpool(x1)
#         x2 = self.Conv2(x2)
# #         print('x2',x2.shape)

#         x3 = wt(x2)
# #         x3 = self.Maxpool(x2)
#         x3 = self.Conv3(x3)
# #         print('x3',x3.shape)

#         x4 = wt(x3)
# #         x4 = self.Maxpool(x3)
#         x4 = self.Conv4(x4)
# #         print('x4',x4.shape)

#         x5 = wt(x4)
# #         x5 = self.Maxpool(x4)
#         x5 = self.Conv5(x5)
# #         print('x5',x5.shape)

#         # decoding + concat path
#         d5=iwt(x5)
# #         d5 = self.Up5(x5)
#         d5 = torch.cat((x4,d5),dim=1)
#         d5 = self.Up_conv5(d5)
# #         print('d5',d5.shape)

#         d4=iwt(d5)       
# #         d4 = self.Up4(d5)
#         d4 = torch.cat((x3,d4),dim=1)
#         d4 = self.Up_conv4(d4)
# #         print('d4',d4.shape)

#         d3=iwt(d4)       
# #         d3 = self.Up3(d4)
#         d3 = torch.cat((x2,d3),dim=1)
#         d3 = self.Up_conv3(d3)
# #         print('d3',d3.shape)

#         d2=iwt(d3)       
# #         d2 = self.Up2(d3)
#         d2 = torch.cat((x1,d2),dim=1)
#         d2 = self.Up_conv2(d2)
# #         print('d2',d2.shape)

#         d1 = self.Conv_1x1(d2)
# #         print('d1',d1.shape)

#         return d1

# class Attwaveletunet_batch(nn.Module):
#     def __init__(self,net_inputch=3,net_outputch=2,num_c=4):
#         super(Attwaveletunet_batch,self).__init__()        
        
#         self.Conv1 = conv_block(ch_in=net_inputch,ch_out=num_c)
#         self.Conv2 = conv_block(ch_in=num_c*4,ch_out=num_c*4)
#         self.Conv3 = conv_block(ch_in=num_c*16,ch_out=num_c*16)
#         self.Conv4 = conv_block(ch_in=num_c*64,ch_out=num_c*64)
#         self.Conv5 = conv_block(ch_in=num_c*256,ch_out=num_c*256)

#         self.Up_conv5 = conv_block(ch_in=num_c*128, ch_out=num_c*64)
#         self.Up_conv4 = conv_block(ch_in=num_c*32, ch_out=num_c*16)
#         self.Up_conv3 = conv_block(ch_in=num_c*8, ch_out=num_c*4)
#         self.Up_conv2 = conv_block(ch_in=num_c*2,ch_out=num_c)
        
#         self.Att5 = Attention_block(F_g=num_c*64,F_l=num_c*64,F_int=num_c*64)
#         self.Att4 = Attention_block(F_g=num_c*16,F_l=num_c*16,F_int=num_c*16)
#         self.Att3 = Attention_block(F_g=num_c*4,F_l=num_c*4,F_int=num_c*4)
#         self.Att2 = Attention_block(F_g=num_c,F_l=num_c,F_int=num_c)
        
#         self.Conv_1x1 = nn.Conv2d(num_c,net_outputch,kernel_size=1,stride=1,padding=0)

#     def forward(self,x):
# #         print('x',x.shape)

#         # encoding path
#         x1 = self.Conv1(x)
# #         print('x1',x1.shape)

#         x2 = wt(x1)
# #         x2 = self.Maxpool(x1)
#         x2 = self.Conv2(x2)
# #         print('x2',x2.shape)

#         x3 = wt(x2)
# #         x3 = self.Maxpool(x2)
#         x3 = self.Conv3(x3)
# #         print('x3',x3.shape)

#         x4 = wt(x3)
# #         x4 = self.Maxpool(x3)
#         x4 = self.Conv4(x4)
# #         print('x4',x4.shape)

#         x5 = wt(x4)
# #         x5 = self.Maxpool(x4)
#         x5 = self.Conv5(x5)
# #         print('x5',x5.shape)

#         # decoding + concat path
#         d5=iwt(x5)
# #         d5 = self.Up5(x5)
#         x4 = self.Att5(g=d5,x=x4)
#         d5 = torch.cat((x4,d5),dim=1)
#         d5 = self.Up_conv5(d5)
# #         print('d5',d5.shape)

#         d4=iwt(d5)       
# #         d4 = self.Up4(d5)
#         x3 = self.Att4(g=d4,x=x3)
#         d4 = torch.cat((x3,d4),dim=1)
#         d4 = self.Up_conv4(d4)
# #         print('d4',d4.shape)

#         d3=iwt(d4)       
# #         d3 = self.Up3(d4)
#         x2 = self.Att3(g=d3,x=x2)
#         d3 = torch.cat((x2,d3),dim=1)
#         d3 = self.Up_conv3(d3)
# #         print('d3',d3.shape)

#         d2=iwt(d3)       
# #         d2 = self.Up2(d3)
#         x1 = self.Att2(g=d2,x=x1)
#         d2 = torch.cat((x1,d2),dim=1)
#         d2 = self.Up_conv2(d2)
# #         print('d2',d2.shape)

#         d1 = self.Conv_1x1(d2)
# #         print('d1',d1.shape)

#         return d1


class waveletunet_base(nn.Module):
    def __init__(self,net_inputch=3,net_outputch=2,num_c=4, Attention=False, RCNN=False, t=2):
        super(waveletunet_base,self).__init__()
        self.Attention = Attention
        self.RCNN = RCNN
        
        self.Conv1 = conv_block(ch_in=net_inputch,ch_out=num_c)
        self.Conv2 = conv_block(ch_in=num_c*4,ch_out=num_c*4)
        self.Conv3 = conv_block(ch_in=num_c*16,ch_out=num_c*16)
        self.Conv4 = conv_block(ch_in=num_c*64,ch_out=num_c*64)
        self.Conv5 = conv_block(ch_in=num_c*256,ch_out=num_c*256)

        self.Up_conv5 = conv_block(ch_in=num_c*128, ch_out=num_c*64)
        self.Up_conv4 = conv_block(ch_in=num_c*32, ch_out=num_c*16)
        self.Up_conv3 = conv_block(ch_in=num_c*8, ch_out=num_c*4)
        self.Up_conv2 = conv_block(ch_in=num_c*2,ch_out=num_c)
        
        if self.Attention:
            self.Att5 = Attention_block(F_g=num_c*64,F_l=num_c*64,F_int=num_c*64)
            self.Att4 = Attention_block(F_g=num_c*16,F_l=num_c*16,F_int=num_c*16)
            self.Att3 = Attention_block(F_g=num_c*4,F_l=num_c*4,F_int=num_c*4)
            self.Att2 = Attention_block(F_g=num_c,F_l=num_c,F_int=num_c)

        if self.RCNN:
            self.RRCNN1 = RRCNN_block(ch_in=net_inputch,ch_out=num_c,t=t)
            self.RRCNN2 = RRCNN_block(ch_in=num_c*4,ch_out=num_c*4,t=t)
            self.RRCNN3 = RRCNN_block(ch_in=num_c*16,ch_out=num_c*16,t=t) 
            self.RRCNN4 = RRCNN_block(ch_in=num_c*64,ch_out=num_c*64,t=t)      
            self.RRCNN5 = RRCNN_block(ch_in=num_c*256,ch_out=num_c*256,t=t)

            self.Up_RRCNN5 = RRCNN_block(ch_in=num_c*128, ch_out=num_c*64,t=t)
            self.Up_RRCNN4 = RRCNN_block(ch_in=num_c*32, ch_out=num_c*16,t=t)
            self.Up_RRCNN3 = RRCNN_block(ch_in=num_c*8, ch_out=num_c*4,t=t)
            self.Up_RRCNN2 = RRCNN_block(ch_in=num_c*2,ch_out=num_c,t=t)

        self.Conv_1x1 = nn.Conv2d(num_c,net_outputch,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
#         print('x',x.shape)

        # encoding path
        x1 = self.Conv1(x) if self.RCNN==False else self.RRCNN1(x)
#         print('x1',x1.shape)

        x2 = wt(x1)
#         x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2) if self.RCNN==False else self.RRCNN2(x2)
#         print('x2',x2.shape)

        x3 = wt(x2)
#         x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3) if self.RCNN==False else self.RRCNN3(x3)
#         print('x3',x3.shape)

        x4 = wt(x3)
#         x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4) if self.RCNN==False else self.RRCNN4(x4)
#         print('x4',x4.shape)

        x5 = wt(x4)
#         x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5) if self.RCNN==False else self.RRCNN5(x5)
#         print('x5',x5.shape)

        # decoding + concat path
        d5=iwt(x5)
#         d5 = self.Up5(d5)
        x4 = self.Att5(g=d5,x=x4) if self.Attention == True else x4
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_conv5(d5) if self.RCNN == False else self.Up_RRCNN5(d5)
#         print('d5',d5.shape)

        d4=iwt(d5)       
#         d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3) if self.Attention == True else x3
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4) if self.RCNN == False else self.Up_RRCNN4(d4)
#         print('d4',d4.shape)

        d3=iwt(d4)       
#         d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2) if self.Attention == True else x2
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3) if self.RCNN == False else self.Up_RRCNN3(d3)
#         print('d3',d3.shape)

        d2=iwt(d3)       
#         d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1) if self.Attention == True else x1
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2) if self.RCNN == False else self.Up_RRCNN2(d2)
#         print('d2',d2.shape)

        d1 = self.Conv_1x1(d2)
#         print('d1',d1.shape)

        return d1

class waveletunet_att(nn.Module):
    def __init__(self,net_inputch=3,net_outputch=2,num_c=4, Attention=False, RCNN=False, t=2):
        super(waveletunet_att,self).__init__()
        
        self.base_net = waveletunet_base(net_inputch=net_inputch,net_outputch=net_outputch,num_c=4, Attention=True, RCNN=False, t=2)
    def forward(self,x):
        yhat = self.base_net(x)
        return yhat

class waveletunet_r2(nn.Module):
    def __init__(self,net_inputch=3,net_outputch=3,num_c=4, Attention=False, RCNN=False, t=2):
        super(waveletunet_r2,self).__init__()
        
        self.base_net = waveletunet_base(net_inputch=net_inputch,net_outputch=net_outputch,num_c=4, Attention=False, RCNN=True, t=2)
    def forward(self,x):
        yhat = self.base_net(x)
        return yhat

class waveletunet_r2att(nn.Module):
    def __init__(self,net_inputch=3,net_outputch=3,num_c=4, Attention=False, RCNN=False, t=2):
        super(waveletunet_r2att,self).__init__()
        
        self.base_net = waveletunet_base(net_inputch=net_inputch,net_outputch=net_outputch,num_c=4, Attention=True, RCNN=True, t=2)
    def forward(self,x):
        yhat = self.base_net(x)
        return yhat
    
# weight standardization
    
class Conv2d(nn.Conv2d):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
#         inplace = True
        inplace = False
        
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=inplace),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=inplace)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
#         inplace = True
        inplace = False

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=inplace)
        )
    
    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
#         inplace = True
        inplace = False
        
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=inplace)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)
        
    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
#         inplace = True
        inplace = False

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=inplace)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
#         inplace= True
        inplace= False

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=inplace)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi  
    

class MCDropout(nn.Dropout):
    def forward(self, input):
        return F.dropout(input, self.p, True, self.inplace)


class U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=2,t=2,norm='batch'):
        super(R2U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t,norm=norm)
        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t,norm=norm)
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t,norm=norm)        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t,norm=norm)        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t,norm=norm)
        
        self.Up5 = up_conv(ch_in=1024,ch_out=512,norm=norm)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t,norm=norm)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256,norm=norm)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t,norm=norm)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128,norm=norm)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t,norm=norm)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64,norm=norm)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t,norm=norm)
        
        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class AttU_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1,norm='instance',mc_dropout=True):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64,norm=norm)
        self.Conv2 = conv_block(ch_in=64,ch_out=128,norm=norm)
        self.Conv3 = conv_block(ch_in=128,ch_out=256,norm=norm)
        self.Conv4 = conv_block(ch_in=256,ch_out=512,norm=norm)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024,norm=norm)

        self.Up5 = up_conv(ch_in=1024,ch_out=512,norm=norm)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256,norm=norm)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512,norm=norm)

        self.Up4 = up_conv(ch_in=512,ch_out=256,norm=norm)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128,norm=norm)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256,norm=norm)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128,norm=norm)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64,norm=norm)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128,norm=norm)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64,norm=norm)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32,norm=norm)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64,norm=norm)
        
        if norm != 'ws':
            self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        else:
            self.Conv_1x1 = Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
            
        if mc_dropout==True:
            self.MCDropout = MCDropout(p=0.3)
        else:
            self.MCDropout = MCDropout(p=0.0)
        
#         self.nnblock512 = NONLocalBlock2D(512)
#         self.nnblock1024 = NONLocalBlock2D(1024)
        
    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)
#         x1 = self.MCDropout(x1)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
#         x2 = self.MCDropout(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
#         x3 = self.MCDropout(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
#         x4 = self.nnblock512(x4) 
#         x4 = self.MCDropout(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
#         x5 = self.nnblock1024(x5) 
        x5 = self.MCDropout(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
#         d5 = self.nnblock512(d5) 
#         d5 = self.MCDropout(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)
#         d4 = self.MCDropout(d4)
        
        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)
#         d3 = self.MCDropout(d3)
        
        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)
#         d2 = self.MCDropout(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class SoftMTL_AttUNet(nn.Module):
    def __init__(self,img_ch=3,output_ch=2,recon_ch=3,norm = 'batch'):
        super(SoftMTL_AttUNet,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64,norm=norm)
        self.Conv2 = conv_block(ch_in=64,ch_out=128,norm=norm)
        self.Conv3 = conv_block(ch_in=128,ch_out=256,norm=norm)
        self.Conv4 = conv_block(ch_in=256,ch_out=512,norm=norm)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024,norm=norm)

        self.Up5 = up_conv(ch_in=1024,ch_out=512,norm=norm)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256,norm=norm)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512,norm=norm)

        self.Up4 = up_conv(ch_in=512,ch_out=256,norm=norm)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128,norm=norm)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256,norm=norm)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128,norm=norm)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64,norm=norm)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128,norm=norm)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64,norm=norm)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32,norm=norm)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64,norm=norm)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        self.Conv_1x1_r = nn.Conv2d(64,recon_ch,kernel_size=1,stride=1,padding=0)
        
        self.MCDropout = MCDropout(p=0.5)
        
        self.nnblock512 = NONLocalBlock2D(512)
        self.nnblock1024 = NONLocalBlock2D(1024)
        
    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)
        x1 = self.MCDropout(x1)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x2 = self.MCDropout(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3 = self.MCDropout(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
#         x4 = self.nnblock512(x4) 
        x4 = self.MCDropout(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
#         x5 = self.nnblock1024(x5) 
        x5 = self.MCDropout(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
#         d5 = self.nnblock512(d5) 
        d5 = self.MCDropout(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)
        d4 = self.MCDropout(d4)
        
        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)
        d3 = self.MCDropout(d3)
        
        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)
        d2 = self.MCDropout(d2)

        d1 = self.Conv_1x1(d2)
    
        # decoding + concat path
        r5 = self.Up5(x5)
        x4 = self.Att5(g=r5,x=x4)
        r5 = torch.cat((x4,r5),dim=1)        
        r5 = self.Up_conv5(r5)
        r5 = self.MCDropout(r5)
        
        r4 = self.Up4(r5)
        x3 = self.Att4(g=r4,x=x3)
        r4 = torch.cat((x3,r4),dim=1)
        r4 = self.Up_conv4(r4)
        r4 = self.MCDropout(r4)
        
        r3 = self.Up3(r4)
        x2 = self.Att3(g=r3,x=x2)
        r3 = torch.cat((x2,r3),dim=1)
        r3 = self.Up_conv3(r3)
        r3 = self.MCDropout(r3)
        
        r2 = self.Up2(r3)
        x1 = self.Att2(g=r2,x=x1)
        r2 = torch.cat((x1,r2),dim=1)
        r2 = self.Up_conv2(r2)
        r2 = self.MCDropout(r2)

        r1 = self.Conv_1x1_r(r2)

        return d1,r1

class HardMTL_AttUNet(nn.Module):
    def __init__(self,img_ch=3,output_ch=2,recon_ch=3,norm = 'batch'):
        super(HardMTL_AttUNet,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64,norm=norm)
        self.Conv2 = conv_block(ch_in=64,ch_out=128,norm=norm)
        self.Conv3 = conv_block(ch_in=128,ch_out=256,norm=norm)
        self.Conv4 = conv_block(ch_in=256,ch_out=512,norm=norm)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024,norm=norm)

        self.Up5 = up_conv(ch_in=1024,ch_out=512,norm=norm)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256,norm=norm)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512,norm=norm)

        self.Up4 = up_conv(ch_in=512,ch_out=256,norm=norm)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128,norm=norm)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256,norm=norm)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128,norm=norm)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64,norm=norm)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128,norm=norm)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64,norm=norm)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32,norm=norm)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64,norm=norm)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        self.Conv_1x1_r = nn.Conv2d(64,recon_ch,kernel_size=1,stride=1,padding=0)
        
        self.MCDropout = MCDropout(p=0.5)
        
        self.nnblock512 = NONLocalBlock2D(512)
        self.nnblock1024 = NONLocalBlock2D(1024)
        
    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)
        x1 = self.MCDropout(x1)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x2 = self.MCDropout(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3 = self.MCDropout(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
#         x4 = self.nnblock512(x4) 
        x4 = self.MCDropout(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
#         x5 = self.nnblock1024(x5) 
        x5 = self.MCDropout(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
#         d5 = self.nnblock512(d5) 
        d5 = self.MCDropout(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)
        d4 = self.MCDropout(d4)
        
        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)
        d3 = self.MCDropout(d3)
        
        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)
        d2 = self.MCDropout(d2)

        d1 = self.Conv_1x1(d2)
        
        r2 = self.Up2(d3)
        x1 = self.Att2(g=r2,x=x1)
        r2 = torch.cat((x1,r2),dim=1)
        r2 = self.Up_conv2(r2)
        r2 = self.MCDropout(r2)

        r1 = self.Conv_1x1_r(r2)

        return d1,r1

class R2AttU_Net(nn.Module):
    def __init__(self,net_inputch=3,net_outputch=2,t=2,norm='batch',p=0):
        super(R2AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=net_inputch,ch_out=64,t=t,norm=norm)
        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t,norm=norm)        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t,norm=norm)        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t,norm=norm)        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t,norm=norm)        

        self.Up5 = up_conv(ch_in=1024,ch_out=512,norm=norm)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256,norm=norm)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t,norm=norm)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256,norm=norm)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128,norm=norm)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t,norm=norm)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128,norm=norm)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64,norm=norm)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t,norm=norm)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64,norm=norm)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32,norm=norm)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t,norm=norm)
        
        self.Conv_1x1 = nn.Conv2d(64,net_outputch,kernel_size=1,stride=1,padding=0)  
        self.MCDropout = MCDropout(p=p)     
        
#         self.nnblock256 = NONLocalBlock2D(256)
#         self.nnblock512 = NONLocalBlock2D(512)
#         self.nnblock1024 = NONLocalBlock2D(1024)

    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)
#         x1 = self.MCDropout(x1)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
#         x2 = self.MCDropout(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)
#         x3 = self.nnblock256(x3) 
#         x3 = self.MCDropout(x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)
#         x4 = self.nnblock512(x4) 
#         x4 = self.MCDropout(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)
#         x5 = self.nnblock1024(x5) 
        x5 = self.MCDropout(x5)
        
        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)        
#         d5 = self.nnblock512(d5) 
#         d5 = self.MCDropout(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)        
#         d4 = self.nnblock256(d4)
#         d4 = self.MCDropout(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)
#         d3 = self.MCDropout(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)
#         d2 = self.MCDropout(d2)

        d1 = self.Conv_1x1(d2)

        return d1

# NNBlock
import torch
from torch import nn
from torch.nn import functional as F

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
#             bn = nn.BatchNorm3d
            bn = nn.GroupNorm
        elif dimension == 2:
#             conv_nd = nn.Conv2d
            conv_nd = Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
#             bn = nn.BatchNorm2d
            bn = nn.GroupNorm
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
#             bn = nn.BatchNorm1d
            bn = nn.GroupNorm

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
#                 bn(self.in_channels)
                bn(int(self.in_channels/16),self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)

class DiscriminateNet(nn.Module):
    def __init__(self, n_class=2):
        super(StanfordBNet, self).__init__()

        self.conv1_1 = nn.Conv2d(n_class, 16, 5, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(3, 8, 5, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(8, 16, 5, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv3_4 = nn.Conv2d(256, 2, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, segmented_input, original_input):
        res1 = self.relu(self.conv1_1(segmented_input))

        res2 = self.relu(self.conv2_1(original_input))
        res2 = F.max_pool2d(res2, 2, stride=1, padding=1)
        res2 = self.relu(self.conv2_2(res2))
        res2 = F.max_pool2d(res2, 2, stride=1, padding=1)

        res3 = torch.cat((res1, res2), 1)

        res3 = self.relu(self.conv3_1(res3))
        res3 = F.max_pool2d(res3, 2, stride=1)
        res3 = self.relu(self.conv3_2(res3))
        res3 = F.max_pool2d(res3, 2, stride=1)
        res3 = self.relu(self.conv3_3(res3))
        res3 = self.conv3_4(res3)
        
        # return res
        out = F.avg_pool2d(res3, (res3.shape[2],res3.shape[3]))
        out = F.softmax(out)
        n , _ , _ ,_  = segmented_input.size()
        return out.view(n,-1).transpose(0,1)[0]