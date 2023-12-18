# StyleGAN2 implementation by BrianTu
# For study purpose only
# Reference: https://github.com/lucidrains/stylegan2-pytorch

# torch modules
import torch
import torch.nn as nn

class BaseModule(nn.Module):
    """
    The base module of StyleGAN2.
    It provides the basic functions to move the model to the target device.
    """
    
    def __init__(self):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
    

    def getDevice(self):
        """
        Get the device of the model.
        """
        return self.dummy_param.device
    
    def getParamNum(self):
        """
        Get the number of parameters of the model.
        """
        return sum(p.numel() for p in self.parameters())