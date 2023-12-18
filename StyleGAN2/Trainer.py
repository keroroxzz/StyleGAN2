# StyleGAN2 implementation by BrianTu
# For study purpose only
# Reference: https://github.com/lucidrains/stylegan2-pytorch
# Partially modified from materials of Machine Learning course by Hung-yi Lee

# import modules
import os
from datetime import datetime
from accelerate import Accelerator
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

# torch module
import torch
import torchvision
from torch.utils.data import DataLoader

from StyleGAN2.StyleGAN2 import StyleGAN2
from StyleGAN2.Loss import SytleGAN2LossFunctions

class TrainerGAN():
    def __init__(self, config):
        self.config = config
        self.device = config["device"]
        self.totalEpoch = config["totalEpoch"]
        self.generatorUpdateFreq = config["generatorUpdateFreq"]
        
        # models
        self.styleGAN2 = StyleGAN2(config)
        self.styleGAN2EMA = StyleGAN2(config)

        self.styleGAN2 = self.styleGAN2.to(self.device)
        self.styleGAN2EMA = self.styleGAN2EMA.to(self.device)
        self.styleGAN2.train()
        self.styleGAN2EMA.eval()
        
        # loss
        self.lossModule = SytleGAN2LossFunctions(config["lossConfig"])
        
        # optimizers and schedulers
        self.optimizerDisctirminator = torch.optim.Adam(
            self.styleGAN2.getDiscriminatorParams(), 
            **self.config["optimizerConfig_discriminator"])
        
        self.optimizerGenerator = torch.optim.Adam(
            self.styleGAN2.getGeneratorParams(),
            **self.config["optimizerConfig_generator"])
        
        schedulerConfig = self.config["schedulerConfig"]
        self.schedulerDiscriminator = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizerDisctirminator, 
            gamma=schedulerConfig['discriminator']['gamma'])
        
        self.schedulerGenerator = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizerGenerator, 
            gamma=schedulerConfig['generator']['gamma'])
        
        # Accelerator is used to accelerate the training process.
        self.accelerator = Accelerator(
            mixed_precision=config['mixed_precision']) \
            if 'mixed_precision' in config.keys() else None
        
        # dataloader and loggers
        self.dataloader = DataLoader(
            self.config["dataset"],
            batch_size=self.config["batch_size"],
            shuffle=True,
            pin_memory=self.config["pin_memory"],
            num_workers=self.config['num_workers'])
        
        # Apply accelerator if fp16 is activated
        if self.accelerator is not None:
            self.stylerModule, self.dataloader, self.optimizerDisctirminator, self.optimizerGenerator =\
            self.accelerator.prepare(self.stylerModule, self.dataloader, self.optimizerDisctirminator, self.optimizerGenerator)
        
        # sources of sample images
        self.z_samples = self.styleGAN2.generatorModule.sampleLatent(100).to(self.device)
        self.noise_samples = self.styleGAN2.generatorModule.sampleImageNoise(100).to(self.device)
        
    def prepareFolders(self):
        
        FORMAT = '%(asctime)s - %(levelname)s: %(message)s'
        logging.basicConfig(level=logging.INFO, 
                            format=FORMAT,
                            datefmt='%Y-%m-%d %H:%M')
        
        self.log_dir = os.path.join(os.path.curdir, 'logs')
        self.ckpt_dir = os.path.join(os.path.curdir, 'checkpoints')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # update dir by time
        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = os.path.join(self.log_dir, time+f'_{self.config["model_type"]}')
        self.ckpt_dir = os.path.join(self.ckpt_dir, time+f'_{self.config["model_type"]}')
        os.makedirs(self.log_dir)
        os.makedirs(self.ckpt_dir)
    
    def emaUpdate(self, src, dst, alpha=0.9):
        """
        Update the model weights by exponential moving average.
        This makes the training process more stable.
        """
        for src_parm, dst_parm in zip(src.parameters(), dst.parameters()):
            dst_parm.data = (1.0-alpha)*src_parm.data + alpha*dst_parm.data
                
    def visualizeSample(self, epoch, save=True):
        
        with torch.no_grad():
            imageSample,_ = self.styleGAN2EMA.generate(self.z_samples, noise=self.noise_samples)
            normImageSample = (imageSample.data + 1) / 2.0

        if save:
            filename = os.path.join(self.log_dir, f'Epoch_{epoch+1:03d}.jpg')
            torchvision.utils.save_image(normImageSample, filename, nrow=10)
            logging.info(f'Save some samples to {filename}.')

        # Show some images during training.
        grid_img = torchvision.utils.make_grid(normImageSample.cpu(), nrow=10)
        plt.figure(figsize=(10,10))
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()
        
    def train(self):
        """
        The training process of the StyleGAN2
        """
        loginfo = {}
        step = 0
        for e, epoch in enumerate(range(self.totalEpoch)):

            progress_bar = tqdm(self.dataloader)
            progress_bar.set_description(f"Epoch {e+1}")

            for i, data in enumerate(progress_bar):
                
                imgs = data.to(self.device)
                batchSize = imgs.size(0)
                
                update_generator = step % self.generatorUpdateFreq == 0

                z = self.styleGAN2.generatorModule.sampleLatent(batchSize)
                
                realImgs = imgs
                
                # Generator forwarding
                if update_generator:
                    fakeImgs,style = self.styleGAN2.generate(z)
                else:
                    with torch.no_grad():
                        fakeImgs,style = self.styleGAN2.generate(z)

                # Discriminator forwarding
                fakeScores = self.styleGAN2.discriminate(fakeImgs)
                
                # Loss
                if update_generator:

                    loss_G = self.lossModule.GeneratorLoss(fakeScores)

                    pathLengthPenalty = self.lossModule.weightedPathLengthRegularization(z, fakeImgs)
                    loss_G = loss_G + pathLengthPenalty
                    loginfo["generator loss"] = loss_G.item()

                else:

                    realScores = self.styleGAN2.discriminate(realImgs)
                    deviationLoss = self.lossModule.weightedDiscriminatorDeviationLoss(realScores)
                    gradientPenalty = self.lossModule.weightedGradientPenalty(self.styleGAN2.discriminatorModule, realImgs, fakeImgs)
                    
                    loss_D = self.lossModule.discriminationLoss(realScores, fakeScores) + gradientPenalty + deviationLoss
                    loss_D = loss_D - self.lossModule.GeneratorLoss(fakeScores)
                    loginfo["discriminator loss"] = loss_D.item()
                    loginfo["gradientPenalty"] = gradientPenalty.item()
                    loginfo["deviationLoss"] = deviationLoss.item()
                    loginfo["pathLengthEMA"] = self.lossModule.pathLengthEMA
                    
                # Backwarding
                if update_generator:

                    self.styleGAN2.generatorModule.zero_grad()
                    if self.accelerator:
                        self.accelerator.backward(loss_G)
                    else:
                        loss_G.backward()
                    
                    self.optimizerGenerator.step()
                    self.schedulerGenerator.step()

                else:

                    self.styleGAN2.discriminatorModule.zero_grad()
                    if self.accelerator:
                        self.accelerator.backward(loss_D)
                    else:
                        loss_D.backward()
                
                    self.optimizerDisctirminator.step()
                    self.schedulerDiscriminator.step()
                    
                if step % 10 == 0:
                    loginfo["lr"] = self.schedulerGenerator.get_lr()
                    progress_bar.set_postfix(**loginfo)
                    self.emaUpdate(self.styleGAN2, self.styleGAN2EMA, alpha=0.9 if epoch>1 else 0.0)
                    
                step += 1
                
            self.visualizeSample(epoch=epoch)
            
            if (e+1) % 5 == 0 or e == 0:
                # Save the checkpoints.
                torch.save(self.styleGAN2.state_dict(), os.path.join(self.ckpt_dir, f'StyleGAN_{e}.pth'))

        logging.info('Finish training')