# StyleGAN2 implementation by BrianTu
# For study purpose only
# Reference: https://github.com/lucidrains/stylegan2-pytorch
# Partially modified from materials of Machine Learning course by Hung-yi Lee

# import modules
import os
import cv2
import glob
import pickle
from tqdm import tqdm
from os.path import exists

# torch module
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, workdirectory, folder, pretransform, posttransform, force_read=False):
        self.posttransform = posttransform
        
        self.workdirectory = workdirectory
        self.files = os.path.join(self.workdirectory, folder)
        self.fnames = glob.glob(os.path.join(self.files, '*'))
        self.packpath = os.path.join(self.workdirectory, 'datapack.pickle')
        
        if (not force_read) and exists(self.packpath):
            print(f'DataPack is found at {self.packpath}, load dataset from the pack...')
            with open(self.packpath, 'rb') as f:
                self.imgs = pickle.load(f)
        else:
            print(f'No DataPack found, load dataset from files...' if not force_read else 'Loading files...')
            self.imgs = []
            for fname in tqdm(self.fnames):
                self.imgs.append(pretransform(cv2.imread(fname,cv2.IMREAD_COLOR)[:,:,::-1]))

            with open(self.packpath, 'wb') as f:
                pickle.dump(self.imgs, f)
        
    def __getitem__(self,idx):
        img = self.imgs[idx]
        img = self.posttransform(img)
        return img

    def __len__(self):
        return len(self.imgs)
    
def makeDatasetWithTransforms(filepath):
    
    pretransform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
    ])
    
    posttransform = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.16, saturation=0.1, hue=0.05),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    
    dataset = ImageDataset(filepath, 'faces', pretransform, posttransform)
    return dataset