# StyleGAN2 implementation by BrianTu
# For study purpose only
# Reference: https://github.com/lucidrains/stylegan2-pytorch
# Partially modified from materials of Machine Learning course by Hung-yi Lee

# import modules
import numpy as np

# torch module
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from StyleGAN2.Discriminator import Discriminator


class SytleGAN2LossFunctions(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pathLengthEMA_decay = self.config['pathLengthRegularization_decay']
        self.pathLengthEMA = 0.0
        
    def weightedGradientPenalty(
            self, 
            discriminator: Discriminator, 
            real_samples: torch.Tensor, 
            fake_samples: torch.Tensor)->torch.Tensor:
        """
        Implement the gradient penalty to promise a stable gradient.
        """

        a = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(real_samples.device)
        samples = (a * real_samples + ((1 - a) * fake_samples)).requires_grad_(True)
        
        discriminatorOutputs = discriminator(samples).view(-1,1)
        
        gradients = torch.autograd.grad(
            outputs=discriminatorOutputs,
            inputs=samples,
            grad_outputs=Variable(torch.ones(real_samples.shape[0], 1), requires_grad=False).to(real_samples.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0].view(real_samples.size(0), -1)
        
        return self.config['GradientPenalty_weight']*((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    def weightedPathLengthRegularization(self, styles: torch.Tensor, images: torch.Tensor)->torch.Tensor:
        """
        Implement the path length regularization in StyleGAN2.
        It encourages linear relationship between the generated image and the style latent.
        """
        device = images.device
        rand_img = torch.randn(images.shape, device=device)
        outputs = (images * rand_img)

        gradients = torch.autograd.grad(
            outputs=outputs,
            inputs=styles,
            grad_outputs=torch.ones(outputs.shape, device=device),
            create_graph=True,
            retain_graph=True
        )
        gradMags = torch.flatten(gradients[0], start_dim=1).square().sum(dim=1).sqrt()
        meanGradMag = gradMags.mean()
        loss = (gradMags-self.pathLengthEMA).square().mean()

        # calculate the moving average of the gradient magnitude
        self.pathLengthEMA = self.pathLengthEMA * self.pathLengthEMA_decay + meanGradMag.item() * (1 - self.pathLengthEMA_decay)

        return self.config['pathLengthRegularization_weight']*loss
                
    def discriminationLoss(self, real: torch.Tensor, fake: torch.Tensor)->torch.Tensor:
        """
        Calculate the hinge loss of the discriminator for both real case and fake case.
        It encourages the discriminator to distinguish the real image from the fake image.
        """
        
        hingLoss = (F.relu(1 - real) + F.relu(1 + fake)).mean()
        return hingLoss
                
    def weightedDiscriminatorDeviationLoss(self, realScore: torch.Tensor)->torch.Tensor:
        """
        Calculate the deviation loss of the discriminator.
        This will encourage the discriminator to produce a stable output.
        """
        
        return self.config['discriminatorDeviationLoss_weight']*realScore.std()
    
    def GeneratorLoss(self, fakeScores: torch.Tensor)->torch.Tensor:
        """
        The generation loss is the negative mean of the fake scores.
        It encourages the generator to produce a better fake image to fool the discriminator.
        """
        
        return -torch.mean(fakeScores)