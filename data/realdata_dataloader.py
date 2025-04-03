from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from data.common import *
from scipy.ndimage import median_filter

def max_value_RealDataset(version):
    return np.load(f'/home/spagnuolo/Desktop/RealData/RealData_{version}/max_value.npy')

class RealDataset(Dataset):
    """RealDataset"""

    def __init__(self, root_dir = '/home/spagnuolo/Desktop/RealData/RealData_',
                 scale=2, mode=0, transform=None, noise = {}, augm = True, erosion = {}, version = '1.0'):
        """
        Args:
            root_dir (string): Directory with all the images.
            scale (float): dataset scale
            mode (int): train = 0 val = 1 test = 2
            transform (callable, optional): Optional transform to be applied on a sample.
            augm (bool): for training, adds rotation and flip
            noise (dict): 'gaussian' for gaussian, 'saltpepper' for salt&pepper
            erosion (dict): 'add_zero_pixels'to add zero pixels before erosion, 'standard_erosion', 'like_gt' for lr[gt<1e-2]=0
            
        """
        self.root_dir = root_dir + version
        self.transform = transform
        self.scale = scale
        self.mode = mode
        self.noise = noise
        self.augm = augm
        self.erosion = erosion
        self.version = version

        # GLOBAL FLAG FOR ITERATION OF PATCHES ON THE SAME IMAGE DURING TRAINING PHASE
        self.iteration_on_image = 0
     
        if self.mode == 0:
            self.images_folder = self.root_dir + '/train_images/train_image_'
            self.depths_hr_folder = self.root_dir + '/train_depths_hr/train_depth_hr_'
            self.depths_lr_folder = self.root_dir + '/train_depths_lr/train_depth_lr_'
        elif self.mode == 1:
            self.images_folder = self.root_dir + '/val_images/val_image_'
            self.depths_hr_folder = self.root_dir + '/val_depths_hr/val_depth_hr_'
            self.depths_lr_folder = self.root_dir + '/val_depths_lr/val_depth_lr_'
        elif self.mode == 2:
            self.images_folder = self.root_dir + '/test_images/test_image_'
            self.depths_hr_folder = self.root_dir + '/test_depths_hr/test_depth_hr_'
            self.depths_lr_folder = self.root_dir + '/test_depths_lr/test_depth_lr_'
            
    def __len__(self):
        if self.mode == 0:
            return 50
        elif self.mode == 1:
            return 10
        else:
            return 0

    def __getitem__(self, idx):
        gt = np.load(self.depths_hr_folder + str(idx) + r'.npy')
        h, w = gt.shape[0], gt.shape[1]
        s = self.scale

        lr = np.load(self.depths_lr_folder + str(idx) + r'.npy')
        lr[lr > 1] = 0

        # LIKE GT
        if self.erosion['like_gt']['apply_training'] == True and self.mode == 0 and random.random() > 0.8:
            gt_down = np.array(Image.fromarray(gt).resize((w//s,h//s), Image.NEAREST))
            lr[gt_down < 1e-2] = 0
        elif self.erosion['like_gt']['apply_val'] == True and self.mode != 0:
            gt_down = np.array(Image.fromarray(gt).resize((w//s,h//s), Image.NEAREST))
            lr[gt_down < 1e-2] = 0

        guidance = np.load(self.images_folder + str(idx) + r'.npy')

        
        if self.mode == 0:
            # VERY BAD CODING, CAN BE IMPROVED, IT WAS THE FASTEST WAY I THOUGHT FOR DOING THIS
            # USES A GLOBAL INDEX TO ITERATE THROUGH IMAGE PATCHES

            h_start, h_end = 65, 1185
            w_start, w_end = 250, 2090
            patch_w, patch_h = 460, 560

            lr = lr[h_start//2:h_end//2, w_start//2:w_end//2]
            gt = gt[h_start:h_end, w_start:w_end]
            guidance = guidance[h_start:h_end, w_start:w_end,:]

            gt = gt[patch_h*(self.iteration_on_image//4) : patch_h*(self.iteration_on_image//4 + 1),
                    patch_w*(self.iteration_on_image%4) :  patch_w*(self.iteration_on_image%4 + 1)]

            guidance = guidance[patch_h*(self.iteration_on_image//4) : patch_h*(self.iteration_on_image//4 + 1),
                patch_w*(self.iteration_on_image%4) :  patch_w*(self.iteration_on_image%4 + 1), :]

            lr = lr[patch_h*(self.iteration_on_image//4) //s: patch_h*(self.iteration_on_image//4 + 1) //s,
                patch_w*(self.iteration_on_image%4) //s:  patch_w*(self.iteration_on_image%4 + 1) //s]

            #print(self.iteration_on_image)
            
            if self.iteration_on_image < 7:
                self.iteration_on_image += 1
            else:
                self.iteration_on_image = 0
            
            gt = np.expand_dims(gt,2)
            lr = np.expand_dims(lr,2)
            if self.augm:
                tmp = augment([guidance, gt, lr])
                guidance, gt, lr = tmp[0], tmp[1], tmp[2]  
            
            #tmp2 = get_patch_realdata([guidance, gt], lr, patch, scale = s)
            #guidance, gt, lr = tmp2[0], tmp2[1], tmp2[2]
            gt = gt.squeeze(-1)
            lr = lr.squeeze(-1)

        ## ADD EROSION
        if self.mode == 0:
            if random.random() > 0.75:
                if self.erosion['add_zero_pixels']['apply'] == True:
                    lr = add_zero_pixels(lr, noise_ratio = 0.001)
                if self.erosion['standard_erosion']['apply'] == True:
                    lr = erosion_augm(lr, kernel = self.erosion['standard_erosion']['kernel'])
        else:
            if self.erosion['add_zero_pixels']['apply'] == True:
                lr = add_zero_pixels(lr, noise_ratio = 0.001)
            if self.erosion['standard_erosion']['apply'] == True:
                lr = erosion_augm(lr, kernel = self.erosion['standard_erosion']['kernel'])

        ## ADD NOISE (GAUSSIAN + SALT&PEPPER)
        if self.mode == 0:
            if random.random() > 0.2:
                if self.noise['gaussian']['apply_training'] == True:
                    lr = add_gaussian_noise(lr, mean = 0, stdv = self.noise['gaussian']['stdv'])
                if self.noise['saltpepper']['apply'] == True:
                     lr = add_salt_pepper(lr, self.mode)
        else:
            if self.noise['gaussian']['apply_val'] == True:
                lr = add_gaussian_noise(lr, mean = 0, stdv = self.noise['gaussian']['stdv'])
            if self.noise['saltpepper']['apply'] == True:
                     lr = add_salt_pepper(lr, self.mode)


        if self.erosion['cutout']['apply_training'] == True and self.mode == 0:
            lr = self.cutout(lr, size = self.erosion['cutout']['size'])
        elif self.erosion['cutout']['apply_val'] == True and self.mode != 0:
            lr = self.cutout(lr, size = self.erosion['cutout']['size'])

        ## TEST WITH DOWN+UP ON LR TO BE MORE SIMILAR TO METAGRASP
        if self.noise['downup_lr']['apply'] == True:
            h_lr, w_lr = lr.shape[0], lr.shape[1]
            lr = np.array(Image.fromarray(lr).resize((w_lr*4,h_lr*4), Image.BICUBIC))
            lr = np.array(Image.fromarray(lr).resize((w_lr,h_lr), Image.NEAREST))
            lr[lr<1e-3]=0

        if self.transform:
            guidance = self.transform(guidance).float()
            gt = self.transform(np.expand_dims(gt,2)).float()
            lr = self.transform(np.expand_dims(lr,2)).float()


        sample = {'guidance': guidance, 'lr': lr, 'gt': gt}
        
        return sample


    def cutout(self, lr, size):
        h, w = lr.shape[0], lr.shape[1]

        x1 = random.randrange(0, (w//2-size))
        x2 = random.randrange(w//2, (w-size))
        x3 = random.randrange(0, (w//2-size))
        x4 = random.randrange(w//2, (w-size))
        
        y1 = random.randrange(0, (h//2-size))
        y2 = random.randrange(0, (h//2-size))
        y3 = random.randrange(h//2, (h-size))    
        y4 = random.randrange(h//2, (h-size))

        lr[y1:y1+size, x1:x1+size] = 0
        lr[y2:y2+size, x2:x2+size] = 0
        lr[y3:y3+size, x3:x3+size] = 0
        lr[y4:y4+size, x4:x4+size] = 0
        
        return lr