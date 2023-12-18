# StyleGAN2 implementation by BrianTu
# For study purpose only
# Reference: https://github.com/lucidrains/stylegan2-pytorch

# import modules
import numpy as np
from math import log2
from utils.perfomanceMeter import pm

# torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# import common modules
from StyleGAN2.BaseModule import BaseModule
from StyleGAN2.CommonModule import LeakyReLU, Blur, AttentionBlock, EqLinear

class StyleMapping(BaseModule):
    """
    The implementation of StyleMapping in StyleGAN2
    It learns the mapping from the style latent space to the intermediate latent space
    """
    def __init__(
            self, 
            latentDim=128, 
            hidden=128, 
            depth=8, 
            lr_mul=0.1):
        
        super().__init__()
        if depth<1:
            raise(f'ERROR: The layer number must be larger than 0. While it is only {depth} layer now.')
        
        layers=[]
        for i in range(depth):
            layers.append(EqLinear(in_dim=hidden if i>0 else latentDim, out_dim=hidden if i<depth-1 else latentDim, lr_mul=lr_mul))
            layers.append(LeakyReLU())
        
        self.mapping = nn.Sequential(*layers)
        
    def forward(self, x, diversity=1.0)->torch.Tensor:
        """
        Mapping the latent vector to the style latent space.
        The diversity can amplify the diversity of generated images.
        """
        
        w = F.normalize(x, dim=1)
        style = self.mapping(w)

        if not diversity==1.0:
            meanStyle = style.mean(dim=0,keepdim=True)
            style = (style-meanStyle)*diversity+meanStyle

        return style

class DemodConv(nn.Module):
    """
    The implementation of Demodulated Convolution in StyleGAN2
    """
    def __init__(
            self, 
            in_dim, 
            out_dim, 
            kernel, 
            demod=True, 
            strid=1, 
            dilation=1, 
            eps=1e-8):
        
        super().__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.demod = demod
        self.eps = eps
        
        self.k = kernel
        self.w = nn.Parameter(torch.randn((out_dim, in_dim, kernel, kernel)))
        nn.init.kaiming_normal_(self.w, a=0, mode='fan_in', nonlinearity='leaky_relu')
    
    def forward(self, x: torch.Tensor, s: torch.Tensor)->torch.Tensor:
        """
        Calculate the demodulated convolution.
        """

        b,_,h,w = x.shape
        
        weight = torch.einsum('ijkl,mj->mijkl', [self.w, s+1])
        
        if self.demod:
            weight = weight*torch.rsqrt((weight**2).sum(dim=(2,3,4), keepdim=True)+self.eps)
            
        weight = weight.view(-1,self.in_dim,self.k,self.k)
        x = F.conv2d(x.reshape(1,-1,h,w), weight, padding='same', groups=b)
        return x.view(-1,self.out_dim,h,w)
    
class ToRGB(nn.Module):
    """
    The implementation of ToRGB block in StyleGAN2
    It learns the transformation from the intermediate feature map to RGB image
    """
    def __init__(
            self, 
            lat_dim, 
            in_dim, 
            upsample, 
            channel=3, 
            mode='bilinear'):
        
        super().__init__()
        self.A = nn.Linear(lat_dim, in_dim)
        self.conv = DemodConv(in_dim, channel, 1, demod=False)
        
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=mode, align_corners=False),
            Blur()
        ) if upsample else None
        
    def forward(self, x, prev, s)->torch.Tensor:

        s=self.A(s)
        x=self.conv(x,s)
        if prev is not None:
            x=x+prev
        if self.upsample is not None:
            x=self.upsample(x)
        return x
        
class GenBlock(nn.Module):
    """
    The generator block in StyleGAN2
    """

    def __init__(
            self, 
            lat_dim, 
            in_dim, 
            out_dim, 
            upsample_mode='bilinear', 
            upsample_input=True, 
            upsample_output=True, 
            channel=3):
        
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2,mode=upsample_mode, align_corners=False) if upsample_input else None
        
        self.A1 = nn.Linear(lat_dim, in_dim)
        self.B1 = nn.Linear(1, out_dim)
        self.conv1 = DemodConv(in_dim, out_dim,3)
        
        self.A2 = nn.Linear(lat_dim, out_dim)
        self.B2 = nn.Linear(1, out_dim)
        self.conv2 = DemodConv(out_dim, out_dim,3)
        
        self.activation = LeakyReLU()
        self.rgb = ToRGB(lat_dim, out_dim, upsample_output, mode=upsample_mode, channel=channel)
        
    def forward(self, x, prev, s, n):
        if self.upsample is not None:
            x=self.upsample(x)
            
        n = n[:, :x.shape[2], :x.shape[3], :]
        n1 = self.B1(n).permute((0,3,2,1))
        n2 = self.B2(n).permute((0,3,2,1))
        
        s1 = self.A1(s)
        x = self.conv1(x,s1)
        x = self.activation(x+n1)
        
        s2 = self.A2(s)
        x = self.conv2(x,s2)
        x = self.activation(x+n2)
        
        rgb = self.rgb(x, prev, s)
        return x, rgb
    
class Generator(BaseModule):
    """
    The Generator of StyleGAN2.
    It learns the transformation from the style latent space to the RGB image.
    """
    
    def __init__(self, 
                 image_size, 
                 styleMapperConfig,
                 styleDim, 
                 attention_layers=[], 
                 nhead=2, 
                 att_ffdim=512, 
                 capacity=16, 
                 const=True, 
                 channel=3, 
                 filter_max=512):
        
        super().__init__()
        
        self.styleDim = styleDim
        self.img_size = image_size
        filters, length = self.getLayerConfig(image_size, capacity, filter_max)
        
        self.attention_layers = attention_layers
        
        self.const = nn.Parameter(torch.randn(1, filters[0], 4, 4)) if const else None
        self.init_conv = nn.Conv2d(filters[0], filters[0], 3, padding='same')
        self.blocks = nn.ModuleList([])
        self.attentions = nn.ModuleList([])

        self.stylerModule = StyleMapping(**styleMapperConfig)
        
        for i in range(length):
            
            self.attentions.append(AttentionBlock(filters[max(i-1,0)], nhead=nhead, ffdim=att_ffdim) if i in self.attention_layers else None)
            
            self.blocks.append(GenBlock(
                styleDim, 
                in_dim = filters[max(i-1,0)], 
                out_dim = filters[i], 
                upsample_mode='bilinear', 
                upsample_input=i>0, 
                upsample_output=i<length-1, 
                channel=channel
            ))
            
    def getLayerConfig(self, img_size: int, capacity: int, filter_max: float)->(np.ndarray, int):
        """
        Decide the layer configuration of the generator given the image size and desired capacity.
        Return: ([filter num,...], layer num)
        """
        
        num = int(log2(img_size)-1)
        filters = np.logspace(num,1,num=num, base=2).astype(int)*capacity
        filters = np.clip(filters, 0, filter_max)
        return filters, num

    def sampleLatent(self, batchSize: int)->torch.Tensor:
        """
        Randomly sample the latent space.
        Return: torch.Tensor[batchSize, latentDim]
        """

        return torch.randn(batchSize, self.styleDim).to(self.getDevice())
    
    def sampleStyle(self, batchSize: int)->torch.Tensor:
        """
        Randomly sample the style latent space.
        Return: torch.Tensor[batchSize, latentDim]
        """

        return torch.randn(batchSize, self.styleDim).to(self.getDevice())
    
    def sampleImageNoise(self, batchSize: int)->torch.Tensor:
        """
        Randomly sample the style latent space.
        Return: torch.Tensor[batchSize, latentDim]
        """

        return torch.randn(batchSize, self.img_size, self.img_size, 1).to(self.getDevice())
            
    def forward(
            self, 
            z:torch.Tensor, 
            noise: torch.Tensor=None, 
            diversity=1.0):
        
        batchSize = z.size(0) 
        
        # generate style latent space
        style = self.stylerModule(z, diversity)
        if noise is None:
            noise = self.sampleImageNoise(style.size(0))

        # generate image
        img = None
        x = self.const.expand(batchSize, -1, -1, -1)
        for g,a in zip(self.blocks, self.attentions):
            if a is not None:
                x = a(x)
            x,img = g(x,img,style,noise)
            
        return img, style