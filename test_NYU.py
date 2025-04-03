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
net.load_state_dict(torch.load("results/saved_models/BEST-NYU-epoch100.pth", map_location=device, weights_only = True))

# SETTINGS FOR NOISE
noise = {'gaussian':{}, 'saltpepper':{}}
noise['gaussian']['apply'] = False
noise['gaussian']['stdv'] = 0.02
noise['saltpepper']['apply'] = False

# SETTINGS FOR EROSION
erosion = {'standard_erosion':{}, 'add_zero_pixels':{}}
erosion['standard_erosion']['apply'] = False
erosion['standard_erosion']['kernel'] = 5
erosion['add_zero_pixels']['apply'] = False

data_transform = transforms.Compose([transforms.ToTensor()])
test_dataset = NYU_v2_dataset(scale=scale, transform=data_transform, mode = 2, noise = noise, augm = True, erosion = erosion, depthColor = 1)
# MINMAX FOR NYU IS STILL IMPLEMENTED USING MIN AND MAX VALUE FOR EACH IMAGE
minmax = load_minmax_NYU(val = True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

with torch.no_grad():
    net.eval()
    rmse = np.zeros(len(test_dataloader))
       
    t = tqdm(iter(test_dataloader), leave=True, total=len(test_dataloader))

    for idx, data in enumerate(t):  
        guidance, lr, gt = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda()
   
            
        out, out_grad, _ = net((guidance, lr))

        rmse[idx] = calc_rmse_raw(gt[0, 0], out[0, 0], minmax[:,idx])

        t.set_description('[test] rmse: %f' % rmse[:idx + 1].mean())
        t.refresh()

    r_mean = rmse.mean()
    print(r_mean)