# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:11:35 2019

@author: scraed
"""
import numpy as np 

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.function import Function
import matplotlib.pyplot as plt

torch.backends.cudnn.deterministic = True


def general_flip(H):
    m = (H-1)//2
    n = H//2
    col = np.arange(0,m+1)
    flipcol = -1*np.flip(np.arange(1,n+1))
    concol = np.concatenate((col,flipcol),0)
    return concol
    
def dist(target):
    '''
    sqrt(m^2 + n^2) in eq(8)
    '''

    _,_,H,W = target.shape
    concol = general_flip(H)
    conrow = general_flip(W)

    m_col = concol[:,np.newaxis] 
    m_row = conrow[np.newaxis,:]
    dist = np.sqrt(m_col*m_col + m_row*m_row) # sqrt(m^2+n^2)
  
    if torch.cuda.is_available():
        dist_ = torch.from_numpy(dist).float().cuda()
    else:
        dist_ = torch.from_numpy(dist).float()
    return dist_

class EnergyLoss(nn.Module):
    def __init__(self,alpha):
        super(EnergyLoss, self).__init__()
        self.energylossfunc = EnergylossFunc.apply
        self.alpha = alpha
        # self.sigma = sigma

    def forward(self,feat,label):
        return self.energylossfunc(feat, label)
    
class EnergyLoss(nn.Module):
    def __init__(self,alpha):
        super(EnergyLoss, self).__init__()
        self.energylossfunc = EnergylossFunc.apply
        self.alpha = alpha

    def forward(self,feat,label):
        return self.energylossfunc(0.5-feat, 0.5-label,self.alpha)
    
class EnergylossFunc(Function):
    '''
    target: ground truth 
    feat: prob of class vessel -0.5, prob of class vessel from softmax output of unet 
    '''
    @staticmethod
    def forward(ctx,feat_levelset,target,alpha):
        target = target.float()
        index_ = dist(target)
        # print(type(index_))
        dim_ = target.shape[1]
        target = torch.squeeze(target,1)
        # G_t + alpha*H(phi) in eq(5)
        dmn = torch.fft.fftn(input =(target + alpha*(feat_levelset)),norm = 'ortho')
        dmn2 = (dmn.abs())**2 #dmn_r * dmn_r + dmn_i * dmn_i # dmn^2
    
        ctx.save_for_backward(feat_levelset,target,dmn,index_)
            
        F_energy = torch.sum(index_*dmn2)/feat_levelset.shape[0]/feat_levelset.shape[1]/feat_levelset.shape[2] # eq(8)
        return F_energy

    @staticmethod
    def backward(ctx,grad_output):
        feature,label,dmn,index_ = ctx.saved_tensors
        index_ = index_.unsqueeze(0).unsqueeze(1)
        F_diff = -0.5*index_*dmn # eq(9) 
        diff = torch.fft.ifftn(input = F_diff, norm = 'ortho').real/feature.shape[0]
        return Variable(-grad_output*diff),None,None,None