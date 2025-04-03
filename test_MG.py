from models.SGNet import *
from models.common_modules import *
from models.GCM import *
from models.SDB import *

from data.nyu_dataloader import *
from data.metagrasp_dataloader import *
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

# USED FOR DIFFERENT OPTIONS DURING EXPERIMENTS
version = '4.2'

device = torch.device('cuda')
net = SGNet(num_feats=n_feats, kernel_size=3, scale=scale, version = version).cuda()
net.load_state_dict(torch.load("results/saved_models/BEST-MG-epoch94.pth", map_location=device, weights_only = True))

# SETTINGS FOR NOISE
noise = {'gaussian':{}, 'saltpepper':{}}
noise['gaussian']['apply'] = False
noise['gaussian']['stdv'] = 0.02
noise['saltpepper']['apply'] = False

# SETTINGS FOR EROSION
erosion = {'standard_erosion':{}, 'add_zero_pixels':{}}
erosion['standard_erosion']['apply'] = False
erosion['standard_erosion']['kernel'] = 11
erosion['add_zero_pixels']['apply'] = False

data_transform = transforms.Compose([transforms.ToTensor()])
test_dataset = MetaGraspDataset(transform=data_transform, mode=2, scale = scale, noise = noise, augm = False, erosion = erosion, version = '2.0')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# MAX VALUE FOR METRIC RESCALING
max_value = torch.from_numpy(load_max_value_MG()).to('cuda')

with torch.no_grad():
    net.eval()
    rmse = np.zeros(len(test_dataloader))
       
    t = tqdm(iter(test_dataloader), leave=True, total=len(test_dataloader))

    for idx, data in enumerate(t):  
        guidance, lr, gt = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda()
   
        crop_h, crop_w = 720, 1120
        
        center_crop_guidance = transforms.CenterCrop(size=(crop_h, crop_w))
        center_crop_gt = transforms.CenterCrop(size=(crop_h, crop_w))
        center_crop_lr = transforms.CenterCrop(size=(crop_h//scale, crop_w//scale))
        guidance = center_crop_guidance(guidance)
        lr = center_crop_lr(lr)
        gt = center_crop_gt(gt)
            
        out, out_grad, _ = net((guidance, lr))

        rmse[idx] = calc_rmse_real(gt[0, 0], out[0, 0], max_value)

        t.set_description('[test] rmse: %f' % rmse[:idx + 1].mean())
        t.refresh()

    r_mean = rmse.mean()
    print(r_mean)