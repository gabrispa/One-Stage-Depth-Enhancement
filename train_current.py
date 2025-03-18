from models.SGNet import *
from models.common_modules import *
from models.GCM import *
from models.SDB import *
from models.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2

from data.nyu_dataloader import *
from data.metagrasp_dataloader import *
from utils import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

scale = 2
dataset_name = 'MG'
version = '4.2'
n_feats = 24
max_epoch = 100

s = datetime.now().strftime('%Y%m%d%H%M')
result_root = 'results/result_train_current/%s-%s-%s-%s' % (s, dataset_name, n_feats, version)
if not os.path.exists(result_root):
    os.mkdir(result_root)

logging.basicConfig(filename='%s/train.log' % result_root, format='%(asctime)s %(message)s', level=logging.INFO)

from torch.utils.tensorboard import SummaryWriter
log_dir = 'runs/%s-%s-%s-%s' % (s, dataset_name, n_feats, version)
writer = SummaryWriter(log_dir=log_dir)

net = SGNet(num_feats=n_feats, kernel_size=3, scale=scale, version = version).cuda()
net_getFrequency = getFrequency()
net_gradient = GetGradientDepth()

criterion = nn.L1Loss()
optimizer = optim.Adam([p for p in net.parameters() if (p.requires_grad == True)], lr=1e-4) 
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5e4, 1e5, 1.6e5], gamma=0.2)
net.train()

data_transform = transforms.Compose([transforms.ToTensor()])
up = nn.Upsample(scale_factor=scale, mode='bicubic')

noise_train = {'gaussian':{}, 'saltpepper':{}}
noise_train['gaussian']['apply'] = True
noise_train['gaussian']['stdv'] = 0.02
noise_train['saltpepper']['apply'] = False

erosion_train = {'standard_erosion':{}, 'add_zero_pixels':{}}
erosion_train['standard_erosion']['apply'] = True
erosion_train['standard_erosion']['kernel'] = 11
erosion_train['add_zero_pixels']['apply'] = True

noise_val = {'gaussian':{}, 'saltpepper':{}}
noise_val['gaussian']['apply'] = False
noise_val['gaussian']['stdv'] = 0.02
noise_val['saltpepper']['apply'] = False

erosion_val = {'standard_erosion':{}, 'add_zero_pixels':{}}
erosion_val['standard_erosion']['apply'] = True
erosion_val['standard_erosion']['kernel'] = 11
erosion_val['add_zero_pixels']['apply'] = False

if dataset_name == 'NYU':
    train_dataset = NYU_v2_dataset(scale=scale, transform=data_transform, mode = 0, noise = noise_train, augm = True, erosion = erosion_train, depthColor = 1)
    val_dataset = NYU_v2_dataset(scale=scale, transform=data_transform, mode = 1, noise = noise_val, augm = False, erosion = erosion_val, depthColor = 1)
    minmax = load_minmax_NYU(val = True)
elif dataset_name == 'MG':
    train_dataset = MetaGraspDataset(scale=scale, transform=data_transform, mode = 0, noise = noise_train, augm = True, erosion = erosion_train, version = '2.0')
    val_dataset = MetaGraspDataset(scale=scale, transform=data_transform, mode = 1, noise = noise_val, augm = False, erosion = erosion_val, version = '2.0')
    #minmax = load_minmax_metagrasp(test = False)
    max_value = torch.from_numpy(load_max_value_MG()).to('cuda')
    
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

if float(version) >= 4.0:
    depth_anything = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    device = torch.device('cuda')
    dav2_path = 'models/DepthAnythingV2/checkpoints/depth_anything_v2_vits.pth'
    depth_anything.load_state_dict(torch.load(dav2_path, map_location=device,weights_only = True))

start_epoch = 0
num_train = len(train_dataloader)
best_rmse = 10.0
best_epoch = 0

for epoch in range(start_epoch, max_epoch):
    net.train()
    running_loss = 0.0

    total_spatial_loss = 0.0
    total_freq_loss = 0.0
    total_grad_loss = 0.0

    t = tqdm(iter(train_dataloader), leave=True, total=len(train_dataloader))
    for idx, data in enumerate(t):
        batches_done = num_train * epoch + idx
        optimizer.zero_grad()
        guidance, lr, gt = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda()

        if float(version) > 0.0:
            
            out, out_grad, preout = net((guidance, lr))
        

            # GRADIENT LOSS
            if float(version) >= 4.0:
                with torch.no_grad():
                    depth_anything.eval()
                    guidance_DA = depth_anything.my_infer_image(guidance)
                    guidance_DA = rescale_guidanceDA(guidance_DA, gt)
                    gt_grad_DA = net_gradient(guidance_DA)

                loss_grad =  dice_loss(gt_grad_DA, out_grad, threshold = 0.01)
            else:
                gt_grad_inp = net_gradient(gt_inpainted_for_grad(gt, dataset_name))
                loss_grad = custom_MAE_gradient(gt_grad_inp, out_grad)
      
            # SPATIAL LOSS
            loss_spa = custom_MAE(gt, out)
            
            # FREQUENCY LOSS
            gt_tiles = split_in_tiles(gt)
            out_tiles = split_in_tiles(out)
            gt_tiles, indexes = discard_bad_tiles(gt_tiles)
            if len(indexes) == 0:
                #print('NO VALID INDEX')
                loss_fre = 0
            else:
                out_tiles = out_tiles[indexes]  
                gt_amps, gt_pha = fourier_transform_tiles(gt_tiles)
                out_amps, out_pha = fourier_transform_tiles(out_tiles)          
                loss_fre_amp, loss_fre_pha = custom_frequency_loss((out_amps, out_pha),(gt_amps, gt_pha))
                loss_fre = 0.5 * loss_fre_amp + 0.5 * loss_fre_pha
            
            if dataset_name == 'NYU':
                gamma_spa = 1
                gamma_grad = 0.005
                gamma_fre = 0.005
            elif dataset_name == 'MG':
                gamma_spa = 1
                gamma_grad = 0.005
                gamma_fre = 0.005
            
            loss = loss_spa*gamma_spa + loss_fre*gamma_fre + loss_grad*gamma_grad
            
        elif version == '0.0':
            out, out_grad, _ = net((guidance, lr))
            out_amp, out_pha = net_getFrequency(out)
            gt_amp, gt_pha = net_getFrequency(gt)
            gt_grad = net_gradient(gt)
            loss_grad1 = criterion(out_grad, gt_grad)
            loss_fre_amp = criterion(out_amp, gt_amp)
            loss_fre_pha = criterion(out_pha, gt_pha)
            loss_fre = 0.5 * loss_fre_amp + 0.5 * loss_fre_pha
            loss_spa = criterion(out, gt)
            loss = loss_spa + 0.002 * loss_fre + 0.001 * loss_grad

        
        loss.backward()

        optimizer.step()
        scheduler.step()
        running_loss += loss.data.item()
        #running_loss_50 = running_loss

        total_spatial_loss += loss_spa
        total_freq_loss += loss_fre
        total_grad_loss += loss_grad
        

        t.set_description('[train epoch:%d] loss: %.8f spa: %.8f grad: %.8f fre: %.8f' % (epoch + 1, running_loss/num_train, loss_spa, loss_grad, loss_fre))
        t.refresh()

    logging.info('epoch:%d iteration:%d running_loss:%.10f' % (epoch + 1, batches_done + 1, running_loss / num_train))

    writer.add_scalar("Loss train current", running_loss/num_train, epoch)
    writer.add_scalars('Losses Extended', {'loss_spa':total_spatial_loss/num_train,
                                    'loss_fre':total_freq_loss/num_train,
                                    'loss_grad': total_grad_loss/num_train}, epoch)

   
    with torch.no_grad():
        net.eval()
        
        rmse = np.zeros(len(val_dataloader))
       
        t = tqdm(iter(val_dataloader), leave=True, total=len(val_dataloader))

        for idx, data in enumerate(t):  
            guidance, lr, gt = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda()

            if dataset_name == 'MG':
                crop_h, crop_w = 720, 1120
                #crop_h, crop_w = 480, 640
                center_crop_guidance = transforms.CenterCrop(size=(crop_h, crop_w))
                center_crop_gt = transforms.CenterCrop(size=(crop_h, crop_w))
                center_crop_lr = transforms.CenterCrop(size=(crop_h//scale, crop_w//scale))
                guidance = center_crop_guidance(guidance)
                lr = center_crop_lr(lr)
                gt = center_crop_gt(gt)
                
            out, out_grad, _ = net((guidance, lr))

            if dataset_name == 'MG':
                rmse[idx] = calc_rmse_real(gt[0, 0], out[0, 0], max_value)
            else:
                rmse[idx] = calc_rmse_raw(gt[0, 0], out[0, 0], minmax[:,idx])
            
            t.set_description('[validate] rmse: %f' % rmse[:idx + 1].mean())
            t.refresh()

        r_mean = rmse.mean()

        writer.add_scalar('Loss val current', r_mean, epoch)
        
        if r_mean < best_rmse:
            best_rmse = r_mean
            best_epoch = epoch
            
            if epoch > 0:
                torch.save(net.state_dict(),
                        os.path.join(result_root, dataset_name + '_BEST_%f_%d.pth' % (best_rmse, best_epoch + 1)))
        logging.info('-------------------------------------------------------')
        logging.info('epoch:%d lr:%f-------mean_rmse:%f (BEST: %f @epoch%d)' % (
            epoch + 1, scheduler.get_last_lr()[0], r_mean, best_rmse, best_epoch + 1))
        logging.info('-------------------------------------------------------')

writer.flush()
