# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:11:35 2019

@author: scraed
"""
import numpy as np 

import torch
import torch.nn as nn
from Unet import *
from train import *
from utility import *
from torch.autograd import Variable
from torch.autograd.function import Function

torch.backends.cudnn.deterministic = True

def odd_flip(H):
    '''
    generate frequency map. 
    
    when height or width of image is odd number,
    creat a array concol [0,1,...,int(H/2)+1,int(H/2),...,0]
    len(concol) = H
    '''
    m = int(H/2)
    col = np.arange(0,m+1)
    flipcol = col[m-1::-1]
    concol = np.concatenate((col,flipcol),0)
    return concol

def even_flip(H):
    '''
    generate frequency map. 
    
    when height or width of image is even number,
    creat a array concol [0,1,...,int(H/2),int(H/2),...,0]
    len(concol) = H
    '''
    m = int(H/2)
    col = np.arange(0,m)
    flipcol = col[m::-1]
    concol = np.concatenate((col,flipcol),0)
    return concol

def dist(target):
    '''
    sqrt(m^2 + n^2) in eq(8)
    '''

    _,_,H,W = target.shape

    if H%2 ==1:
        concol = odd_flip(H)
    else:
        concol = even_flip(H)
        
    if W%2 == 1:
        conrow = odd_flip(W)
    else:
        conrow = even_flip(W)
        
    m_col = concol[:,np.newaxis] 
    m_row = conrow[np.newaxis,:]
    dist = np.sqrt(m_col*m_col + m_row*m_row) # sqrt(m^2+n^2)
  
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        dist_ = torch.from_numpy(dist).float().cuda()
    else:
        dist_ = torch.from_numpy(dist).float()
    return dist_

class EnergyLoss(nn.Module):
    def __init__(self,cuda,alpha,sigma):
        super(EnergyLoss, self).__init__()
        self.energylossfunc = EnergylossFunc.apply
        self.alpha = alpha
        self.cuda = cuda
        self.sigma = sigma

    def forward(self,feat,label):
        return self.energylossfunc(self.cuda,feat, label,self.alpha,self.sigma)
    
class EnergylossFunc(Function):
    '''
    target: ground truth 
    feat: Z -0.5. Z：prob of your target class(here is vessel) with shape[B,1,H,W]. 
    Z from softmax output of unet with shape [B,C,H,W]. C: number of classes
    alpha: default 0.35
    sigma: default 0.25
    '''
    @staticmethod
    def forward(ctx,cuda,feat,target,alpha,sigma,Gaussian = False):
        hardtanh = nn.Hardtanh(min_val=0, max_val=1, inplace=False)
        target = target.float()
        index_ = dist(target)
        dim_ = target.shape[1]
        target = torch.squeeze(target,1)
        I1 = target + alpha*hardtanh(feat/sigma) # G_t + alpha*H(phi) in eq(5)
        dmn = torch.rfft(I1,2,normalized = True, onesided = False)
        dmn_r = dmn[:,:,:,0] # dmn's real part
        dmn_i = dmn[:,:,:,1] # dmm's imagine part
        dmn2 = dmn_r * dmn_r + dmn_i * dmn_i # dmn^2

        ctx.save_for_backward(feat,target,dmn,index_)
            
        F_energy = torch.sum(index_*dmn2)/feat.shape[0]/feat.shape[1]/feat.shape[2] # eq(8)
        
        return F_energy

    @staticmethod
    def backward(ctx,grad_output):
        feature,label,dmn,index_ = ctx.saved_tensors
        index_ = torch.unsqueeze(index_,0)
        index_ = torch.unsqueeze(index_,3)
        F_diff = -0.5*index_*dmn # eq(9) 
        diff = torch.irfft(F_diff,2,normalized = True, onesided = False)/feature.shape[0] # eq
        return None,Variable(-grad_output*diff),None,None,None
