# StyleGAN2 implementation by BrianTu
# For study purpose only
# Reference: https://github.com/lucidrains/stylegan2-pytorch
# Partially modified from materials of Machine Learning course by Hung-yi Lee

# import modules
import os
import cv2
from typing import Iterator

# torch module
import torch

from StyleGAN2.BaseModule import BaseModule
from StyleGAN2.Generator import Generator
from StyleGAN2.Discriminator import Discriminator

class StyleGAN2(BaseModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        
        self.generatorModule = Generator(**self.config["generatorConfig"])
        self.discriminatorModule = Discriminator(**self.config["discriminatorConfig"])

    def getImageSize(self)->int:
        """
        Get the generation image size.
        """
        return self.generatorModule.img_size

    def getGeneratorParams(self)->Iterator[torch.nn.Parameter]:
        """
        Get the parameters of the generator.
        """
        return self.generatorModule.parameters()

    def getDiscriminatorParams(self)->Iterator[torch.nn.Parameter]:
        """
        Get the parameters of the discriminator.
        """
        return self.discriminatorModule.parameters()

    def generate(
            self, 
            latent: torch.Tensor, 
            noise: torch.Tensor=None, 
            diversity: float=1.0)->(torch.Tensor, torch.Tensor):
        """
        Generate images from latent space with noises.
        Retrun: (images, styles)
        """
        return self.generatorModule(latent, noise, diversity)

    def discriminate(self, images: torch.Tensor)->torch.Tensor:
        """
        Discriminate the images.
        """
        return self.discriminatorModule(images)

    def inference(
            self, 
            n_generate=1000, 
            batch=50, 
            n_output=30, 
            diversity=0.7, 
            dir='output', 
            quality=90)->None:
        """
        Inference the generation model and save the result.
        """
        i=1
        self.eval()

        while True:

            z = self.generatorModule.sampleLatent(batch)
            imgs,_ = self.generate(z, diversity) 
            imgs = 255.0 * (imgs.data + 1) / 2.0  # normalization to 0..255

            # write to files
            os.makedirs(dir, exist_ok=True)
            for n in range(imgs.size(0)):
                cv2.imwrite(f'{dir}/{i}.jpg', imgs[n].permute(1,2,0).cpu().detach().numpy()[:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                i+=1
                if i>1000:
                    break
            if i>1000:
                break