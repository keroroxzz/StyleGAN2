# StyleGAN2 implementation by BrianTu
# For study purpose only
# Reference: https://github.com/lucidrains/stylegan2-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from utils.perfomanceMeter import pm

def LeakyReLU(p=0.2, inplace=True):
    return nn.LeakyReLU(p, inplace=inplace)

class Blur(nn.Module):
    """
    The warpper of GaussianBlur in torchvision for StyleGAN2
    """
    def __init__(self, kernel=[0.25, 0.5, 0.25]):
        super().__init__()
        self.blur = transforms.GaussianBlur(3, sigma=(1.0, 1.0))
        
    def forward(self, x):
        return self.blur(x)

class EqLinear(nn.Module):
    """
    The implementation of equalized linear layer in StyleGAN2
    It repleace the linear layer which can scale the learning rate of the weights
    """
    def __init__(self, in_dim, out_dim, lr_mul=0.1, bias=True):
        super().__init__()
        self.w = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.b = nn.Parameter(torch.zeros(out_dim))
        else:
            self.b = None
        self.lr_mul = lr_mul
    
    def forward(self, x):
        x = F.linear(
            x, 
            self.w*self.lr_mul, 
            bias=self.b*self.lr_mul if self.b is not None else None)
        return x
    
class AttentionBlock(nn.Module):
    """
    The implementation of attention block in StyleGAN2
    """

    def __init__(self, dim, nhead, num_encoder_layers=1, num_decoder_layers=1, ffdim=512, dropout=0.1):
        super().__init__()
        self.atten = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead, 
            dim_feedforward=ffdim, 
            dropout=dropout,
            layer_norm_eps=1e-05, 
            batch_first=False, 
            norm_first=True)
    
    def forward(self, x):
        b,c,h,w = x.size()
        x = x.transpose(1,3)
        x = x.reshape(b,w*h,c)
        x = x.transpose(0,1)
        x = self.atten(x)
        x = x.transpose(0,1)
        x = x.reshape(b,w,h,c)
        x = x.transpose(1,3)
        return x