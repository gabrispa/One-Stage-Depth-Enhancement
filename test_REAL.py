from models.SGNet import *
from models.common_modules import *
from models.GCM import *
from models.SDB import *

from data.realdata_dataloader import *
from utils import *
import matplotlib.pyplot as plt
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from importlib import reload
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

scale = 2
n_feats = 24
version = '4.3'
dataset_v = '1.2'
device = torch.device('cuda')
net = SGNet(num_feats=n_feats, kernel_size=3, scale=scale, version = version).cuda()
net.load_state_dict(torch.load("results/result_train_current/20250270044-REAL-24-4.2/REAL_BEST_1.753700_61.pth", map_location=device, weights_only = True))

noise= {'gaussian':{}, 'saltpepper':{}, 'downup_lr':{}}
noise['gaussian']['apply'] = False
noise['gaussian']['stdv'] = 0.02
noise['saltpepper']['apply'] = False
noise['downup_lr']['apply'] = False

erosion = {'standard_erosion':{}, 'like_gt':{}, 'add_zero_pixels':{}, 'cutout':{}}
erosion['standard_erosion']['apply'] = False
erosion['standard_erosion']['kernel'] = 5
erosion['like_gt']['apply_training'] = False
erosion['like_gt']['apply_val'] = True
erosion['add_zero_pixels']['apply'] = False
erosion['cutout']['apply'] = True
erosion['cutout']['size'] = 64


data_transform = transforms.Compose([transforms.ToTensor()])
test_dataset = RealDataset(scale=scale, transform=data_transform, mode = 1, noise = noise, augm = True, erosion = erosion, version = dataset_v)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
max_value = torch.from_numpy(max_value_RealDataset(version = dataset_v)).to('cuda')*100.0

with torch.no_grad():
    net.eval()
    rmse = np.zeros(len(test_dataloader))
       
    t = tqdm(iter(test_dataloader), leave=True, total=len(test_dataloader))

    for idx, data in enumerate(t):  
        guidance, lr, gt = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda()
      
        #crop_h, crop_w = 720, 1120
        #center_crop_guidance = transforms.CenterCrop(size=(crop_h, crop_w))
        #center_crop_gt = transforms.CenterCrop(size=(crop_h, crop_w))
        #center_crop_lr = transforms.CenterCrop(size=(crop_h//scale, crop_w//scale))
        #guidance = center_crop_guidance(guidance)
        #lr = center_crop_lr(lr)
        #gt = center_crop_gt(gt)
            
        lr = lr[:,:,100:700,250:850]
        gt = gt[:,:,100*2:700*2,250*2:850*2]
        guidance = guidance[:,:,100*2:700*2,250*2:850*2]

        resize1 = transforms.Resize((150,150), interpolation = InterpolationMode.NEAREST)
        resize2 = transforms.Resize((300,300), interpolation = InterpolationMode.NEAREST)
        resize3 = transforms.Resize((300,300), interpolation = InterpolationMode.BILINEAR)
        lr = resize1(lr)
        gt = resize2(gt)
        guidance = resize3(guidance)

        out, out_grad, _ = net((guidance, lr))
        rmse[idx] = calc_rmse_real(gt[0, 0], out[0, 0], max_value)

        # Save out images
        #if (idx % 5) == 0:
        #    np.save(f'saved_images/2025-02-09/out_{idx}.npy', out[0,0].cpu().numpy())
            #Image.fromarray((out[0,0].cpu().numpy() * 255).astype(np.uint8)).save(f'saved_images/out_{str(idx)}.png')
        
        t.set_description('[validate] rmse: %f' % rmse[:idx + 1].mean())
        t.refresh()

    r_mean = rmse.mean()
    print(r_mean)