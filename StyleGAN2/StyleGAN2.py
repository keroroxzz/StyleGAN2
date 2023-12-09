# StyleGAN2 implementation by BrianTu
# For study purpose only
# Reference: https://github.com/lucidrains/stylegan2-pytorch
# Partially modified from materials of Machine Learning course by Hung-yi Lee

# import module
import os
from datetime import datetime

# import module
import torch
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from math import log2, sqrt

# third party modules
from accelerate import Accelerator
import matplotlib.pyplot as plt
import numpy as np
import logging
import cv2
from tqdm import tqdm

from StyleGAN2.Generator import Generator, StyleMapping
from StyleGAN2.Discriminator import Discriminator
from utils.perfomanceMeter import pm

class TrainerGAN():
    def __init__(self, config):
        self.config = config
        
        self.accelerator = Accelerator(mixed_precision=config['mixed_precision']) if 'mixed_precision' in config.keys() else None
        
        self.S = StyleMapping(
            in_dim=self.config["latent_dim"],
            hidden=self.config["style_mapping_hidden"],
            depth=self.config["style_mapping_layers"],
            lr_mul=self.config["style_mapping_lr_mul"])
        
        self.G = Generator(
            img_size = self.config["image_size"],
            lat_dim = self.config["latent_dim"],
            attention_layers = self.config["Gen_attention_layers"],
            capacity = self.config["Gen_capacity"],
            fmap_max = self.config["Gen_filter_max"],
            nhead = self.config["Gen_nhead"],
            att_ffdim = self.config["Gen_att_ffdim"])
        
        self.D = Discriminator(
            image_size = self.config["image_size"],
            capacity = self.config["Dsc_capacity"],
            attn_layers = self.config["Dsc_attention_layers"],
            fmap_max = self.config["Dsc_filter_max"],
            nhead = self.config["Dsc_nhead"],
            att_ffdim = self.config["Dsc_att_ffdim"])
        
        self.S_ema = StyleMapping(
            in_dim=self.config["latent_dim"],
            hidden=self.config["style_mapping_hidden"],
            depth=self.config["style_mapping_layers"],
            lr_mul=self.config["style_mapping_lr_mul"])
        
        self.G_ema = Generator(
            img_size = self.config["image_size"],
            lat_dim = self.config["latent_dim"],
            attention_layers = self.config["Gen_attention_layers"],
            capacity = self.config["Gen_capacity"],
            fmap_max = self.config["Gen_filter_max"],
            nhead = self.config["Gen_nhead"],
            att_ffdim = self.config["Gen_att_ffdim"])
        
        self.opt_D = torch.optim.Adam(
            self.D.parameters(),
            lr=self.config["lr"],
            betas=self.config["betas"])
        
        self.opt_G = torch.optim.Adam(
            list(self.G.parameters())+list(self.S.parameters()),
            lr=self.config["lr"],
            betas=self.config["betas"])
        
        self.scheduler_D = torch.optim.lr_scheduler.ExponentialLR(self.opt_D, gamma=self.config['gamma'])
        self.scheduler_G = torch.optim.lr_scheduler.ExponentialLR(self.opt_G, gamma=self.config['gamma'])
        
        self.dataloader = None
        self.log_dir = os.path.join(os.path.curdir, 'logs')
        self.ckpt_dir = os.path.join(os.path.curdir, 'checkpoints')
        
        FORMAT = '%(asctime)s - %(levelname)s: %(message)s'
        logging.basicConfig(level=logging.INFO, 
                            format=FORMAT,
                            datefmt='%Y-%m-%d %H:%M')
        
        self.steps = 0
        self.z_samples = Variable(torch.randn(100, self.config["latent_dim"])).cuda()
        self.noise_samples = Variable(torch.randn(100, self.G.img_size, self.G.img_size, 1)).cuda()
        
    def prepare_environment(self, mydataset):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        # update dir by time
        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = os.path.join(self.log_dir, time+f'_{self.config["model_type"]}')
        self.ckpt_dir = os.path.join(self.ckpt_dir, time+f'_{self.config["model_type"]}')
        os.makedirs(self.log_dir)
        os.makedirs(self.ckpt_dir)
        
        # create dataset by the above function
        self.dataloader = DataLoader(
            mydataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            pin_memory=self.config["pin_memory"],
            num_workers=self.config['num_workers'])
        
        # model preparation
        self.S = self.S.cuda()
        self.G = self.G.cuda()
        self.D = self.D.cuda()
        self.S_ema = self.S_ema.cuda()
        self.G_ema = self.G_ema.cuda()
        self.S.train()
        self.G.train()
        self.D.train()
        self.S_ema.eval()
        self.G_ema.eval()
        
        # Show the numbers of parameters
        print(f'Generator has totally {sum(p.numel() for p in self.G.parameters())} params.')
        print(f'Discriminator has totally {sum(p.numel() for p in self.D.parameters())} params.')
        
        # Apply accelerator if fp16 is activated
        if self.accelerator is not None:
            self.S, self.G, self.D, self.dataloader, self.opt_D, self.opt_G =\
            self.accelerator.prepare(self.S, self.G, self.D, self.dataloader, self.opt_D, self.opt_G)
        
    def gradientPenalty(self, real_samples, fake_samples):
        a = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).cuda()
        samples = (a * real_samples + ((1 - a) * fake_samples)).requires_grad_(True)
        
        d_samples = self.D(samples).view(-1,1)
        
        gradients = torch.autograd.grad(
            outputs=d_samples,
            inputs=samples,
            grad_outputs=Variable(torch.ones(real_samples.shape[0], 1), requires_grad=False).cuda(),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0].view(real_samples.size(0), -1)
        
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    def pl_lengths(styles, images):
        """
        Not Used for now.
        This is the only part that is not implemented yet for StyleGAN2.
        But it still works without this part.
        """
        device = images.device
        num_pixels = images.shape[2] * images.shape[3]
        pl_noise = torch.randn(images.shape, device=device) / sqrt(num_pixels)
        outputs = (images * pl_noise).sum()

        # return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()
    
    def emaUpdate(self, src, dst, alpha=0.9):
        """
        Update the model weights by exponential moving average.
        This makes the training process more stable.
        """
        for src_parm, dst_parm in zip(src.parameters(), dst.parameters()):
            dst_parm.data = (1.0-alpha)*src_parm.data + alpha*dst_parm.data
    
    def getAverageScore(self):
        self.G.eval()
        self.D.eval()
        with torch.no_grad():
            for data in tqdm(self.dataloader):
                imgs = data.cuda()
                bs = imgs.size(0)

                z = Variable(torch.randn(bs, self.config["latent_dim"])).cuda()
                r_imgs = imgs
                f_imgs = self.G(z)

                r_score = self.D(r_imgs).item()
                f_score = self.D(f_imgs).item()
                print(r_score)
                
    def visualizeSample(self, epoch, save=True):
        
        with torch.no_grad():
            style_sample = self.S_ema(self.z_samples)
            f_imgs_sample = (self.G_ema(style_sample, self.noise_samples).data + 1) / 2.0

        if save:
            filename = os.path.join(self.log_dir, f'Epoch_{epoch+1:03d}.jpg')
            torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
            logging.info(f'Save some samples to {filename}.')

        # Show some images during training.
        grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
        plt.figure(figsize=(10,10))
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()
                
    def DLoss(self, real, fake, gp, rstd):
        return (F.relu(1 - real) + F.relu(1 + fake)).mean() + self.config['gp_amp']*gp + self.config['std_amp']*rstd
    
    def GLoss(self, fake):
        return -fake
        
    def train(self):
        for e, epoch in enumerate(range(self.config["n_epoch"])):
            progress_bar = tqdm(self.dataloader)
            progress_bar.set_description(f"Epoch {e+1}")
            for i, data in enumerate(progress_bar):
                
                imgs = data.cuda()
                bs = imgs.size(0)
                
                update_generator=self.steps % self.config["n_critic"] == 0

                z = Variable(torch.randn(bs, self.config["latent_dim"])).cuda()
                noise = Variable(torch.randn(bs, self.G.img_size, self.G.img_size, 1)).cuda()
                
                r_imgs = imgs
                
                if update_generator:
                    style = self.S(z)
                    f_imgs = self.G(style, noise)
                else:
                    with torch.no_grad():
                        style = self.S(z)
                        f_imgs = self.G(style, noise)

                # Discriminator forwarding
                r_raw = self.D(r_imgs)
                f_raw = self.D(f_imgs)
                
                # Loss
                r_std = torch.std(r_raw)
                gradient_penalty = self.gradientPenalty(r_imgs, f_imgs)
                
                loss_D = self.DLoss(r_raw, f_raw, gradient_penalty, r_std)
                if update_generator:
                    f_score = torch.mean(f_raw)
                    loss_D = loss_D - self.GLoss(f_score)
                    loss_G = self.GLoss(f_score)
                    
                # Backwarding
                self.D.zero_grad()
                
                if self.accelerator is not None:
                    self.accelerator.backward(loss_D, retain_graph=update_generator)
                else:
                    loss_D.backward(retain_graph=update_generator)
                
                if update_generator:
                    self.S.zero_grad()
                    self.G.zero_grad()
                    if self.accelerator:
                        self.accelerator.backward(loss_G)
                    else:
                        loss_G.backward()
                    
                    self.opt_G.step()
                
                self.opt_D.step()
                
                self.scheduler_D.step()
                self.scheduler_G.step()
                    
                if self.steps % 10 == 0:
                    progress_bar.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item(), lr=self.scheduler_G.get_lr())
                    self.emaUpdate(self.G, self.G_ema, alpha=0.9 if epoch>1 else 0.0)
                    self.emaUpdate(self.S, self.S_ema, alpha=0.9 if epoch>1 else 0.0)
                    
                self.steps += 1
                
            self.visualizeSample(epoch=epoch)
            
            if (e+1) % 5 == 0 or e == 0:
                # Save the checkpoints.
                torch.save(self.S.state_dict(), os.path.join(self.ckpt_dir, f'S_{e}.pth'))
                torch.save(self.G.state_dict(), os.path.join(self.ckpt_dir, f'G_{e}.pth'))
                torch.save(self.D.state_dict(), os.path.join(self.ckpt_dir, f'D_{e}.pth'))
                torch.save(self.S_ema.state_dict(), os.path.join(self.ckpt_dir, f'Sema_{e}.pth'))
                torch.save(self.G_ema.state_dict(), os.path.join(self.ckpt_dir, f'Gema_{e}.pth'))

        logging.info('Finish training')

    def inference(self, n_generate=1000, batch=50, n_output=30, truncation=0.7, dir='output', quality=90):
        i=1
        self.S.eval()
        self.G.eval()


        while True:

            z = Variable(torch.randn(batch, self.config["latent_dim"])).cuda()
            w_avg = self.S(z).mean(dim=0,keepdim=True)[:batch]
            noise = Variable(torch.randn(batch, self.G.img_size, self.G.img_size, 1)).cuda()*1.0

            w = self.S_ema(z)
            if truncation<1.0:
                w = (w-w_avg)*truncation+w_avg

            imgs = 255.0 * (self.G_ema(w,noise).data + 1) / 2.0

            os.makedirs(dir, exist_ok=True)
            for n in range(imgs.size(0)):
                cv2.imwrite(f'{dir}/{i}.jpg', imgs[n].permute(1,2,0).cpu().detach().numpy()[:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                i+=1
                if i>1000:
                    break
            if i>1000:
                break