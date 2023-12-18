# StyleGAN2 implementation by BrianTu
# For study purpose only
# Reference: https://github.com/lucidrains/stylegan2-pytorch

# import modules
import numpy as np
from math import log2

# torch modules
import torch
import torch.nn as nn

# import common modules
from StyleGAN2.BaseModule import BaseModule
from StyleGAN2.CommonModule import LeakyReLU, Blur, AttentionBlock

class DiscriminatorBlock(nn.Module):
    """
    The Discriminator Block of StyleGAN2.
    """

    def __init__(
            self, 
            input_channels, 
            filters, 
            downsample=True):
        
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            LeakyReLU(),
            nn.Conv2d(filters, filters, 3, padding=1),
            LeakyReLU()
        )

        self.downsample = nn.Sequential(
            Blur(),
            nn.Conv2d(filters, filters, 3, padding = 1, stride = 2)
        ) if downsample else None

    def forward(self, x: torch.Tensor)->torch.Tensor:
        res = self.conv_res(x)
        x = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return (x + res) * (1.0/np.sqrt(2))

class Discriminator(BaseModule):
    """
    The Discriminator of StyleGAN2.
    It learns to distinguish the real image from the fake image.
    """

    def __init__(
            self, 
            image_size, 
            capacity = 16, 
            attention_layers = [], 
            filter_max = 512, 
            nhead=2, 
            att_ffdim=512):
        
        super().__init__()

        filters, num_layers = self.getLayerConfig(image_size, capacity, filter_max)
        
        self.blocks = nn.ModuleList([])
        self.attentions = nn.ModuleList([])

        for i in range(num_layers):

            self.blocks.append(
                DiscriminatorBlock(
                    filters[max(i-1,0)],
                    filters[i],
                    downsample=i<num_layers-1))

            self.attentions.append(AttentionBlock(filters[i], nhead=nhead, ffdim=att_ffdim) if i in attention_layers else None)

        self.final_conv = nn.Conv2d(filters[-1], filters[-1], 3, padding=1)
        self.flatten = nn.Flatten()
        self.to_logit = nn.Linear(2 * 2 * filters[-1], 1)

    def getLayerConfig(self, img_size: int, capacity: int, filter_max: float)->(np.ndarray, int):
        """
        Decide the layer configuration of the generator given the image size and desired capacity.
        Return: ([filter num,...], layer num)
        """
        
        num = int(log2(img_size))
        filters = np.logspace(0,num,num=num+1, base=2).astype(int)*capacity*4
        filters = np.clip(filters, 0, filter_max)
        filters[0] = 3
        return filters, num
    
    def forward(self, x: torch.Tensor)->torch.Tensor:

        for (block, attn_block) in zip(self.blocks, self.attentions):
            x = block(x)

            if attn_block is not None:
                x = attn_block(x)

        x = self.final_conv(x)
        x = self.flatten(x)
        x = self.to_logit(x)
        return x.squeeze()