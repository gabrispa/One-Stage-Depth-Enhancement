from models.SGNet import *
from models.common_modules import *
from models.GCM import *
from models.SDB import *
from models.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2

from data.nyu_dataloader import *
from data.metagrasp_dataloader import *
from data.realdata_dataloader import *
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
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda')

scale = 2
dataset_name = 'REAL'

# USED FOR DIFFERENT OPTIONS DURING EXPERIMENTS
version = '4.2'

n_feats = 24
max_epoch = 200
dataset_v = '1.2'

# LOGGING CONFIG
s = datetime.now().strftime('%Y%m%d%H%M')
result_root = 'results/result_train_current/%s-%s-%s-%s' % (s, dataset_name, n_feats, version)
if not os.path.exists(result_root):
    os.mkdir(result_root)
logging.basicConfig(filename='%s/train.log' % result_root, format='%(asctime)s %(message)s', level=logging.INFO)


# TENSORBOARD IMPORT
from torch.utils.tensorboard import SummaryWriter
log_dir = 'runs/%s-%s-%s-%s' % (s, dataset_name, n_feats, version)
writer = SummaryWriter(log_dir=log_dir)


# NET CREATION AND LOADING PARAMETERS
net = SGNet(num_feats=n_feats, kernel_size=3, scale=scale, version = version).cuda()
net.load_state_dict(torch.load("results/result_train_current/202503082257-MG-24-4.2/MG_BEST_0.318149_94.pth", map_location=device, weights_only = True))
net_getFrequency = getFrequency()
net_gradient = GetGradientDepth()

criterion = nn.L1Loss()
optimizer = optim.Adam([p for p in net.parameters() if (p.requires_grad == True)], lr=1e-6) 
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5e4, 1e5, 1.6e5], gamma=0.2)
net.train()

data_transform = transforms.Compose([transforms.ToTensor()])
up = nn.Upsample(scale_factor=scale, mode='bicubic')

# NOISE SETTING --> FINAL TRAINING NOISE = FALSE TO ALL 
noise= {'gaussian':{}, 'saltpepper':{}, 'downup_lr':{}}
noise['gaussian']['apply_training'] = False
noise['gaussian']['apply_val'] = False
noise['gaussian']['stdv'] = 0.02
noise['saltpepper']['apply'] = False
noise['downup_lr']['apply'] = False


# EROSION SETTINGS -->  FINAL TRAINING NOISE = ONLY LIKE_GT WITH PROBABILITY 20%
erosion = {'standard_erosion':{}, 'like_gt':{}, 'add_zero_pixels':{}, 'cutout':{}}
erosion['standard_erosion']['apply'] = False
erosion['standard_erosion']['kernel'] = 5
erosion['like_gt']['apply_training'] = True
erosion['like_gt']['apply_val'] = False
erosion['add_zero_pixels']['apply'] = False
erosion['cutout']['apply_training'] = False
erosion['cutout']['apply_val'] = False
erosion['cutout']['size'] = 24

# DATASETS CREATION
train_dataset = RealDataset(scale=scale, transform=data_transform, mode = 0, noise = noise, augm = True, erosion = erosion, version = dataset_v)
val_dataset = RealDataset(scale=scale, transform=data_transform, mode = 1, noise = noise, augm = False, erosion = erosion, version = dataset_v)
max_value = torch.from_numpy(max_value_RealDataset(version = dataset_v)).to('cuda')*100.0
    
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle = True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

# LOAD DAV2 FOR GRADIENT LOSS AFTER
if float(version) >= 4.0:
    depth_anything = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    device = torch.device('cuda')
    dav2_path = 'models/DepthAnythingV2/checkpoints/depth_anything_v2_vits.pth'
    depth_anything.load_state_dict(torch.load(dav2_path, map_location=device,weights_only = True))

start_epoch = 0
num_images = len(train_dataloader)
num_train = num_images * 8
best_rmse = 10.0
best_epoch = 0

for epoch in range(start_epoch, max_epoch):
    net.train()
    running_loss = 0.0

    total_spatial_loss = 0.0
    total_freq_loss = 0.0
    total_grad_loss = 0.0


    #t = tqdm(iter(train_dataloader), leave=True, total=len(train_dataloader))
    t = tqdm(range(num_images))
    for idx in t:
        batches_done = num_train * epoch + idx * 8
        optimizer.zero_grad()

        # FOR EACH IMAGE ITERATE THROUGH 8 PATCHES (THIS IS MANAGED IN THE realdata_dataloader.py USING A GLOBAL VARIABLE)
        # BAD CODING, TO BE IMPROVED.
        for i in range(8):
            data = train_dataset[idx]
            guidance, lr, gt = data['guidance'].unsqueeze(0).cuda(), data['lr'].unsqueeze(0).cuda(), data['gt'].unsqueeze(0).cuda()

            # VERSION > 0.0 --> OUR VERSIONS
            if float(version) > 0.0:            
                out, out_grad, preout = net((guidance, lr))

                # GRADIENT LOSS
                if float(version) >= 4.0:
                    with torch.no_grad():
                        depth_anything.eval()
                        guidance_DA = depth_anything.my_infer_image(guidance)
                        guidance_DA = rescale_guidanceDA(guidance_DA, gt)
                        gt_grad_DA = net_gradient(guidance_DA)

                    loss_grad =  dice_loss(gt_grad_DA, out_grad, threshold = 0.0075)
        
                # SPATIAL LOSS
                if float(version) >= 4.2:
                    loss_spa = custom_spatial_loss(gt, out, threshold = 0.0071)
                else:
                    loss_spa = custom_MAE(gt, out)
        
                # FREQUENCY LOSS
                gt_tiles = split_in_tiles(gt)
                out_tiles = split_in_tiles(out)
                gt_tiles, indexes = discard_bad_tiles(gt_tiles)
                if len(indexes) == 0:
                    loss_fre = 0
                else:
                    out_tiles = out_tiles[indexes]   
                    gt_amps, gt_pha = fourier_transform_tiles(gt_tiles)
                    out_amps, out_pha = fourier_transform_tiles(out_tiles)         
                    loss_fre_amp, loss_fre_pha = custom_frequency_loss((out_amps, out_pha),(gt_amps, gt_pha))
                    loss_fre = 0.5 * loss_fre_amp + 0.5 * loss_fre_pha
                
                
                gamma_spa = 0.005      # SPATIAL
                gamma_grad = 0.005      # GRADIENT
                gamma_fre = 0.005   # FREQUENCY 
                
                loss = gamma_spa*loss_spa + gamma_fre*loss_fre + loss_grad*gamma_grad 
                
                
                '''plt.figure()
                f, axarr = plt.subplots(2,3, figsize=(30, 15)) 

                axarr[0,0].imshow(guidance[0].detach().cpu().permute(1,2,0).numpy())
                axarr[0,0].set_title('GUIDANCE')

                im1 = axarr[0, 1].imshow((lr[0, 0]*75).detach().cpu().numpy())  # Grayscale
                axarr[0,1].set_title('LR')
                step = 50  
                axarr[0,1].set_xticks(np.arange(0, lr.shape[3], step))
                axarr[0,1].set_yticks(np.arange(0, lr.shape[2], step))
                axarr[0,1].grid(color='black', linestyle='--', linewidth=0.2)
                f.colorbar(im1, ax=axarr[0,1])  # Add colorbar

                im2 = axarr[1, 0].imshow((gt[0, 0]*75).detach().cpu().numpy())  # Grayscale
                axarr[1, 0].set_title('GT')
                step = 100  # Grid spacing (every 100 pixels)
                axarr[1,0].set_xticks(np.arange(0, gt.shape[3], step))
                axarr[1,0].set_yticks(np.arange(0, gt.shape[2], step))
                axarr[1,0].grid(color='black', linestyle='--', linewidth=0.2)
                f.colorbar(im2, ax=axarr[1, 0])  # Add colorbar

                out[0,0,0,0] = 0
                im3 = axarr[1, 1].imshow((out[0, 0]*75).detach().cpu().numpy())  # Grayscale
                axarr[1, 1].set_title('OUTPUT')
                step = 100  
                axarr[1,1].set_xticks(np.arange(0, out.shape[3], step))
                axarr[1,1].set_yticks(np.arange(0, out.shape[2], step))
                axarr[1,1].grid(color='black', linestyle='--', linewidth=0.2)
                f.colorbar(im3, ax=axarr[1, 1])  # Add colorbar

                im4 = axarr[0, 2].imshow((out_grad[0, 0]).detach().cpu().numpy())  # Grayscale
                axarr[0,2].set_title('OUTPUT GRADIENT')
                step = 100  
                axarr[0,2].set_xticks(np.arange(0, out_grad.shape[3], step))
                axarr[0,2].set_yticks(np.arange(0, out_grad.shape[2], step))
                axarr[0,2].grid(color='black', linestyle='--', linewidth=0.2)
                f.colorbar(im4, ax=axarr[0, 2])  # Add colorbar

                im4 = axarr[1, 2].imshow((gt_grad_DA[0, 0]).detach().cpu().numpy() > 0.01)  # Grayscale
                axarr[1, 2].set_title('DA GRADIENT')
                step = 100  
                axarr[1,2].set_xticks(np.arange(0, gt_grad_DA.shape[3], step))
                axarr[1,2].set_yticks(np.arange(0, gt_grad_DA.shape[2], step))
                axarr[1,2].grid(color='black', linestyle='--', linewidth=0.2)
                f.colorbar(im4, ax=axarr[1, 2])  # Add colorbar

                plt.savefig(f'./saved_images/tmp_train/tmp_train_{epoch+1}.png')
                plt.close()'''
                
            # VERSION 0.0 = BASE SGNET MODEL
            elif version == '0.0':
                out_amp, out_pha = net_getFrequency(out)
                gt_amp, gt_pha = net_getFrequency(gt)
                gt_grad = net_gradient(gt)
                loss_grad1 = criterion(out_grad, gt_grad)
                loss_fre_amp = criterion(out_amp, gt_amp)
                loss_fre_pha = criterion(out_pha, gt_pha)
                loss_fre = 0.5 * loss_fre_amp + 0.5 * loss_fre_pha
                loss_spa = criterion(out, gt)
                loss = loss_spa + 0.002 * loss_fre + 0.001 * loss_grad1

            # BACKPROP      
            loss.backward()

            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            #running_loss_50 = running_loss
            
            t.set_description('[train epoch:%d] loss: %.8f spa: %.8f grad: %.8f fre: %.8f' % (epoch + 1, running_loss/num_train, loss_spa, loss_grad, loss_fre))
            t.refresh()

            total_spatial_loss += loss_spa
            total_freq_loss += loss_fre
            total_grad_loss += loss_grad
        

    logging.info('epoch:%d iteration:%d running_loss:%.10f' % (epoch + 1, batches_done + 1, running_loss / num_train))

    writer.add_scalar('Loss train current', running_loss/num_train, epoch)
    writer.add_scalars('Losses Extended', {'loss_spa':total_spatial_loss/num_train,
                                    'loss_fre':total_freq_loss/num_train,
                                    'loss_grad': total_grad_loss/num_train}, epoch)

    # VALIDATION
    with torch.no_grad():
        net.eval()
        
        rmse = np.zeros(len(val_dataloader))
       
        t = tqdm(iter(val_dataloader), leave=True, total=len(val_dataloader))

        for idx, data in enumerate(t):  
            guidance, lr, gt = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda()

            # CENTER CROP FOR MEMORY ISSUES
            crop_h, crop_w = 720, 1120
            center_crop_guidance = transforms.CenterCrop(size=(crop_h, crop_w))
            center_crop_gt = transforms.CenterCrop(size=(crop_h, crop_w))
            center_crop_lr = transforms.CenterCrop(size=(crop_h//scale, crop_w//scale))
            guidance = center_crop_guidance(guidance)
            lr = center_crop_lr(lr)
            gt = center_crop_gt(gt)
                
            # OUTPUT
            out, out_grad, _ = net((guidance, lr))

            # RMSE
            rmse[idx] = calc_rmse_real(gt[0, 0], out[0, 0], max_value)

            # SOME PLOTTING
            if idx == 0:
                plt.figure()
                f, axarr = plt.subplots(2,3, figsize=(30, 15)) 

                axarr[0,0].imshow(guidance[0].detach().cpu().permute(1,2,0).numpy())
                axarr[0,0].set_title('GUIDANCE')

                im1 = axarr[0, 1].imshow((lr[0, 0]*75).detach().cpu().numpy())  # Grayscale
                axarr[0,1].set_title('LR')
                step = 50  
                axarr[0,1].set_xticks(np.arange(0, lr.shape[3], step))
                axarr[0,1].set_yticks(np.arange(0, lr.shape[2], step))
                axarr[0,1].grid(color='black', linestyle='--', linewidth=0.2)
                f.colorbar(im1, ax=axarr[0,1])  # Add colorbar

                im2 = axarr[1, 0].imshow((gt[0, 0]*75).detach().cpu().numpy())  # Grayscale
                axarr[1, 0].set_title('GT')
                step = 100  # Grid spacing (every 100 pixels)
                axarr[1,0].set_xticks(np.arange(0, gt.shape[3], step))
                axarr[1,0].set_yticks(np.arange(0, gt.shape[2], step))
                axarr[1,0].grid(color='black', linestyle='--', linewidth=0.2)
                f.colorbar(im2, ax=axarr[1, 0])  # Add colorbar

                out[0,0,0,0] = 0
                im3 = axarr[1, 1].imshow((out[0, 0]*75).detach().cpu().numpy())  # Grayscale
                axarr[1, 1].set_title('OUTPUT')
                step = 100  
                axarr[1,1].set_xticks(np.arange(0, out.shape[3], step))
                axarr[1,1].set_yticks(np.arange(0, out.shape[2], step))
                axarr[1,1].grid(color='black', linestyle='--', linewidth=0.2)
                f.colorbar(im3, ax=axarr[1, 1])  # Add colorbar

                im3 = axarr[0,2].imshow((out_grad[0, 0]).detach().cpu().numpy())  # Grayscale
                axarr[0,2].set_title('OUT GRADIENT')
                step = 100  
                axarr[0,2].set_xticks(np.arange(0, out_grad.shape[3], step))
                axarr[0,2].set_yticks(np.arange(0, out_grad.shape[2], step))
                axarr[0,2].grid(color='black', linestyle='--', linewidth=0.2)
                f.colorbar(im3, ax=axarr[0, 2])  # Add colorbar

                plt.savefig(f'./saved_images/tmp_val4/tmp_val_{epoch+1}.png')
                plt.close()
                
            t.set_description('[validate] rmse: %f' % rmse[:idx + 1].mean())
            t.refresh()

        r_mean = rmse.mean()

        writer.add_scalar('Loss val current', r_mean, epoch)
        
        # SAVE BEST EPOCH
        if r_mean < best_rmse:
            best_rmse = r_mean
            best_epoch = epoch
            if epoch > 0:
                torch.save(net.state_dict(),
                           os.path.join(result_root, dataset_name + "_BEST_%f_%d.pth" % (best_rmse, best_epoch + 1)))
        logging.info('-------------------------------------------------------')
        logging.info('epoch:%d lr:%f-------mean_rmse:%f (BEST: %f @epoch%d)' % (
            epoch + 1, scheduler.get_last_lr()[0], r_mean, best_rmse, best_epoch + 1))
        logging.info('-------------------------------------------------------')
    

writer.flush()