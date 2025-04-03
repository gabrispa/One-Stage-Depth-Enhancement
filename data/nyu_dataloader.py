from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from data.common import *
import random

# RETRIEVE AN ARRAY WITH MINMAX FOR EACH IMAGE
def load_minmax_NYU(test = False, val = False, train = False):
    minmax = np.load('/home/spagnuolo/Desktop/NYU_v2_dataset_3.0/NYU_minmax.npy', allow_pickle = True)
    train_minmax = minmax.tolist()['train_minmax']
    test_minmax = minmax.tolist()['test_minmax']
    val_minmax = minmax.tolist()['val_minmax']
    
    if test:
        return test_minmax 
    elif train:
        return train_minmax
    elif val:
        return val_minmax

class NYU_v2_dataset(Dataset):
    """NYUDataset_v2"""

    def __init__(self, root_dir = '/home/spagnuolo/Desktop/NYU_v2_dataset_3.0',
                 scale=4, mode=0, transform=None,  noise = {}, augm = True, erosion = {}, depthColor = 0):
        """
        Args:
            root_dir (string): Directory with all the images.
            scale (float): dataset scale
            mode (int): train = 0 val = 1 test = 2
            noise (bool): Add salt&pepper and gaussian noise
            augm (bool): Add augmentation in training (rotation and flip)
            erosion (bool): Add erosion to LR
            transform (callable, optional): Optional transform to be applied on a sample.
            depthColor (int): 0 = rawDepth, 1 = missing values applied to colorDepth, 2 = colorDepth
        """
        self.root_dir = root_dir
        self.transform = transform
        self.scale = scale
        self.mode = mode
        self.noise = noise
        self.augm = augm
        self.erosion = erosion
        self.depthColor = depthColor
        
        if self.mode == 0:
            self.images_folder = root_dir + '/train_images/train_image_'
            if self.depthColor == 1:
                self.depths_folder = root_dir + '/train_depthsRawC/train_depthC_'
            elif self.depthColor == 2:
                self.depths_folder = root_dir + '/train_depthsColor/train_depthColor_'
            else:
                self.depths_folder = root_dir + '/train_depthsRaw/train_depth_'
        elif self.mode == 1:
            self.images_folder = root_dir + '/val_images/val_image_'
            if self.depthColor == 1:
                self.depths_folder = root_dir + '/val_depthsRawC/val_depthC_'
            elif self.depthColor == 2:
                self.depths_folder = root_dir + '/val_depthsColor/val_depthColor_'
            else:
                self.depths_folder = root_dir + '/val_depthsRaw/val_depth_'
        elif self.mode == 2:
            self.images_folder = root_dir + '/test_images/test_image_'
            if self.depthColor == 1:
                self.depths_folder = root_dir + '/test_depthsRawC/test_depthC_'
            elif self.depthColor == 2:
                self.depths_folder = root_dir + '/test_depthsColor/test_depthColor_'
            else:
                self.depths_folder = root_dir + '/test_depthsRaw/test_depth_'
    
    def __len__(self):
        if self.mode == 0:
            return 1000
        elif self.mode == 1:
            return 400
        else:
            return 48

    def __getitem__(self, idx):
        gt = np.load(self.depths_folder + str(idx) + r'.npy')
        guidance = np.load(self.images_folder + str(idx) + r'.npy')
        
        s = self.scale
        h, w = gt.shape[0], gt.shape[1]

        if self.mode == 0:
            patch = 256
            h, w = patch, patch
            gt = np.expand_dims(gt,2)
            
            if self.augm:
                tmp = augment([guidance, gt])
                guidance, gt= tmp[0], tmp[1]
            tmp2 = get_patch([guidance, gt], patch)
            guidance, gt = tmp2[0], tmp2[1]
            
            gt = gt.squeeze(-1)

        if self.depthColor == 2:
            gt = add_big_noise(gt, noise_ratio = 0.35)

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
        
        # CREATE LR ONLY AFTER EROSION, TO HAVE SMOOTH EDGES IN ERODED PARTS
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
