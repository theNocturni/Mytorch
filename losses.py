import torch
import torch.nn as nn
import numpy as np
import monai
from monai.networks.utils import one_hot

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self,yhat,y):
        y = y[:,0].long()
        loss = self.ce(yhat,y)
        return loss 

class DiceCELoss(nn.Module):
    def __init__(self):        
        super(DiceCELoss, self).__init__()
        self.dice = monai.losses.GeneralizedDiceLoss(softmax=False, to_onehot_y=True)
        self.ce = CrossEntropyLoss()        
        
    def forward(self,yhat,y):
        dice = self.dice(yhat,y)
        ce = self.ce(yhat,y)
        return dice+ce

class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """
#     def __init__(self, theta0=3, theta=3, alpha = 0.7, gamma = 0.75): #DRIVE
    def __init__(self, theta0=3, theta=15, alpha = 0.7, gamma = 0.75):  #AMC
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-batch
        """
        
        n, c, _, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
#         pred = torch.softmax(pred, dim=1)

        # one-hot vector of ground truth
        one_hot_gt = one_hot(gt, c)

        # boundary map
        gt_b = F.max_pool2d(1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)
        pred_b_ext = F.max_pool2d(pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)
        
        # to check hyper-parameter
        idx= 0
        print('boundary_loss')
        print(torch.unique(gt_b),torch.unique(gt_b_ext))
        plt.figure(figsize=(24,8))
        plt.subplot(161)
        plt.imshow(gt[idx,0].cpu().detach().numpy())
        plt.subplot(162)
        plt.imshow(gt_b[idx,0].cpu().detach().numpy())
        plt.subplot(163)
        plt.imshow(gt_b_ext[0,idx].cpu().detach().numpy())
        plt.subplot(164)
        plt.imshow(pred[idx,1].cpu().detach().numpy())
        plt.subplot(165)
        plt.imshow(pred_b[idx,0].cpu().detach().numpy())
        plt.subplot(166)
        plt.imshow(pred_b_ext[idx,0].cpu().detach().numpy())
        plt.show()
        
        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

# #         original impliment
#         # Precision, Recall
#         P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
#         R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)
#         # Boundary F1 Score
#         BF1 = 2 * P * R / (P + R + 1e-7)
#         # summing BF1 Score for each class and average over mini-batch
#         loss = torch.mean(1 - BF1)

        # my impliment
        # Precision, Recall
        TP = torch.sum(pred_b * gt_b, dim=2) 
        TN = torch.sum(pred_b_ext * gt_b_ext, dim=2)
        FN = torch.sum(pred_b_ext * gt_b, dim=2)
        FP = torch.sum(pred_b * gt_b_ext, dim=2)
        ALL = TP+TN+FN+FP
        
        smooth = 1e-7

        TV = (TP/ALL + smooth)/(TP/ALL + self.alpha*FN/ALL + (1-self.alpha)*FP/ALL + smooth)
#         loss = torch.mean(1 - TV)
        loss = torch.mean(torch.pow(1 - TV, self.gamma))

        return loss

from sklearn.metrics import *
import numpy as np
import cv2
import torch

def soft_skeletonize(x, thresh_width=20):
    '''
    Differenciable aproximation of morphological skelitonization operaton
    thresh_width - maximal expected width of vessel
    '''

    for i in range(thresh_width):
        min_pool_x = torch.nn.functional.max_pool2d(x*-1, (3, 3), 1, 1)*-1
        contour = torch.nn.functional.relu(torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1) - min_pool_x)
        x = torch.nn.functional.relu(x - contour)
    return x

def norm_intersection(center_line, vessel):
    '''
    inputs shape  (batch, channel, height, width)
    intersection formalized by first ares
    x - suppose to be centerline of vessel (pred or gt) and y - is vessel (pred or gt)
    '''
    smooth = 1.
#     clf = center_line.view(*center_line.shape[:2], -1)
#     vf = vessel.view(*vessel.shape[:2], -1)
    clf = center_line.flatten()
    vf = vessel.flatten()
    intersection = (clf * vf).sum(-1)
    return (intersection + smooth) / (clf.sum(-1) + smooth)

def soft_cldice_loss(pred, target, target_skeleton=None):
    '''
    inputs shape  (batch, channel, height, width).
    calculate clDice loss
    Because pred and target at moment of loss calculation will be a torch tensors
    it is preferable to calculate target_skeleton on the step of batch forming,
    when it will be in numpy array format by means of opencv
    '''
    
    pred = torch.argmax(pred,1).unsqueeze(1).float()
    cl_pred = soft_skeletonize(pred)
    if target_skeleton is None:
        target_skeleton = soft_skeletonize(target)
        
#     print('soft_cldice_loss')
#     plt.figure(figsize=(24,24))
#     plt.subplot(141)
#     plt.imshow(target[0,0].cpu().detach())
#     plt.subplot(142)
#     plt.imshow(target_skeleton[0,0].cpu().detach())
#     plt.subplot(143)
#     plt.imshow(pred[0,0].cpu().detach())
#     plt.subplot(144)
#     plt.imshow(cl_pred[0,0].cpu().detach())
#     plt.show()    
    
    iflat = norm_intersection(cl_pred, target)
    tflat = norm_intersection(target_skeleton, pred)
    intersection = iflat * tflat
    loss = 1-((2. * intersection) / (iflat + tflat))
    return loss

class clDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        return soft_cldice_loss(pred,gt).squeeze()

def softmax_T(tensor,T=1):
    return F.softmax(tensor/T,1)

class BoundaryCELoss(nn.Module):
    def __init__(self):        
        super(BoundaryCELoss, self).__init__()
        self.ce = CrossEntropyLoss()
        self.boundary = BoundaryLoss()

    def forward(self,yhat,y):
        ce = self.ce(yhat,y) 
        boundary = self.boundary(yhat,y)        
        return ce+boundary


class clDiceCELoss(nn.Module):
    def __init__(self):        
        super(clDiceCELoss, self).__init__()
        self.ce = CrossEntropyLoss()
        self.cldice = clDiceLoss()

    def forward(self,yhat,y):
        cldice = self.cldice(yhat,y)
        ce = self.ce(yhat,y)     
        return ce+cldice