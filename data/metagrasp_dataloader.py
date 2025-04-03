from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from data.common import *

# DEPRECATED; USED FOR METAGRASP 1.0, WHERE MINMAX RESCALING WAS DONE USING SINGLE IMAGE MINMAX VALUES
def load_minmax_metagrasp(test = False):
    root = '/home/spagnuolo/Desktop/MetaGraspDataset/MG_1.0'
    if test:
        minmax = np.load(root + '/test_depths/test_minmax.npy',allow_pickle = True)
    else:
        minmax = np.load(root + '/val_depths/val_minmax.npy',allow_pickle = True)
    return minmax

# RETRIEVE GLOBAL MAXIMUM FOR METRIC RESCALING 
def load_max_value_MG():
    return np.load('/home/spagnuolo/Desktop/MetaGraspDataset/MG_2.0/max_value_MG.npy')

class MetaGraspDataset(Dataset):
    """MetaGraspDataset"""

    def __init__(self, scale=8, mode=0, transform=None, noise = {}, augm = True, erosion = {}, version = '2.0'):
        """
        Args:
            root_dir (string): Directory with all the images.
            scale (float): dataset scale
            mode (int): train = 0 val = 1 test = 2
            transform (callable, optional): Optional transform to be applied on a sample.
            noise (dict): 
            augm (bool):
            erosion (dict):
        """
        self.root_dir = f'/home/spagnuolo/Desktop/MetaGraspDataset/MG_{version}'
        self.transform = transform
        self.scale = scale
        self.mode = mode
        self.noise = noise
        self.augm = augm
        self.erosion = erosion
        self.version = version
        self.lenghts = {
            '1.0': {'train': 2000, 'val': 300, 'test': 29},
            '2.0': {'train': 1929, 'val': 300, 'test': 100}
        }
     
        if self.mode == 0:
            self.images_folder = self.root_dir + '/train_images/train_image_'
            self.depths_folder = self.root_dir + '/train_depths/train_depth_'
        elif self.mode == 1:
            self.images_folder = self.root_dir + '/val_images/val_image_'
            self.depths_folder = self.root_dir + '/val_depths/val_depth_'
        elif self.mode == 2:
            self.images_folder = self.root_dir + '/test_images/test_image_'
            self.depths_folder = self.root_dir + '/test_depths/test_depth_'
            
    def __len__(self):
        if self.mode == 0:
            return self.lenghts[self.version]['train']
        elif self.mode == 1:
            return self.lenghts[self.version]['val']
        else:
            return self.lenghts[self.version]['test']

    def __getitem__(self, idx):
        gt = np.load(self.depths_folder + str(idx) + r'.npy')
        if self.version == '1.0':
            gt[gt == -1] = 0
            gt[gt >= 0.95] = 0
        guidance = np.load(self.images_folder + str(idx) + r'.npy')
        h, w = gt.shape[0], gt.shape[1]
        s = self.scale
        
        if self.mode == 0:
            patch = 256
            h, w = patch, patch
            gt = np.expand_dims(gt,2)
            if self.augm:
                tmp = augment([guidance, gt])
                guidance, gt = tmp[0], tmp[1]
            tmp2 = get_patch_metagrasp([guidance, gt], patch)
            guidance, gt = tmp2[0], tmp2[1]
            gt = gt.squeeze(-1)

        gt2 = gt.copy()
        ## ADD EROSION
        if self.mode == 0:
            if random.random() > 0.75:
                if self.erosion['add_zero_pixels']['apply'] == True:
                    gt2 = add_zero_pixels(gt2, noise_ratio = 0.001)
                if self.erosion['standard_erosion']['apply'] == True:
                    gt2 = erosion_augm(gt2, kernel = self.erosion['standard_erosion']['kernel'])
        else:
            if self.erosion['add_zero_pixels']['apply'] == True:
                gt2 = add_zero_pixels(gt2, noise_ratio = 0.001)
            if self.erosion['standard_erosion']['apply'] == True:
                gt2 = erosion_augm(gt2, kernel = self.erosion['standard_erosion']['kernel'])

        # CREATE LR
        lr = np.array(Image.fromarray(gt2).resize((w//s,h//s), Image.BICUBIC))
        lr[lr<1e-2] = 0
        

        ## ADD NOISE (GAUSSIAN + SALT&PEPPER)
        if self.mode == 0:
            if random.random() > 0.2:
                if self.noise['gaussian']['apply'] == True:
                    lr = add_gaussian_noise(lr, mean = 0, stdv = self.noise['gaussian']['stdv'])
                if self.noise['saltpepper']['apply'] == True:
                     lr = add_salt_pepper(lr, self.mode)
        else:
            if self.noise['gaussian']['apply'] == True:
                lr = add_gaussian_noise(lr, mean = 0, stdv = self.noise['gaussian']['stdv'])
            if self.noise['saltpepper']['apply'] == True:
                     lr = add_salt_pepper(lr, self.mode)

        lr[lr<1e-2] = 0
  
        if self.transform:
            guidance = self.transform(guidance).float()#.unsqueeze(0)
            gt = self.transform(np.expand_dims(gt,2)).float()#.unsqueeze(0)
            lr = self.transform(np.expand_dims(lr,2)).float()#.unsqueeze(0)


        sample = {'guidance': guidance, 'lr': lr, 'gt': gt}
        
        return sample
