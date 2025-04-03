import torch
import cv2 as cv
import numpy as np
import torch
from models.common_modules import *
from data.common import *
import torch.nn as nn
import torch.nn.functional as F

# SPATIAL LOSS, MAE ACCOUNTING FOR GT > 0 VALUES
def custom_MAE(gt,out):
    return torch.mean(torch.abs(gt[gt>0] - out[gt>0]))
  
# SPATIAL LOSS USED IN FINE TUNING TO WEIGH MORE ERRORS GREATER THAN 0.5 cm
def custom_spatial_loss(gt, out, threshold):
    # MASK1 = error < 0.5cm (corresponds to 0.0.0071 after rescaling)
    mask1 = (torch.abs(gt - out) <= threshold) & (gt > 0)
    loss1 = torch.mean(torch.abs(gt[mask1] - out[mask1]))

    # MASK2 = error > 0.5cm
    mask2 = (torch.abs(gt - out) > threshold) & (gt > 0)
    loss2 = torch.mean(torch.abs(gt[mask2] - out[mask2])) * 100

    return torch.nan_to_num(loss1) + torch.nan_to_num(loss2)

## INPAINT GT BEFORE CALCULATING THE GRADIENT (USED IN PREVIOUS VERSIONS INSTEAD OF DEPTH ANYTHING)
def gt_inpainted_for_grad(gt, dataset_name):
    x = gt.clone()
    if dataset_name == 'MG' or dataset_name == 'REAL':
        mask = (x == 0)[0,0].int().cpu().numpy()
    elif dataset_name == 'NYU':
        mask = (x < 0.01)[0,0].int().cpu().numpy()
    mask = (mask*255).astype(np.uint8)
    x = x[0,0].cpu().numpy()
    x = torch.from_numpy(cv.inpaint(x, mask, 1 ,cv.INPAINT_NS)).unsqueeze(0).unsqueeze(0).cuda()
    return x
    
def custom_MAE_gradient(a,b, threshold = 0.0):
    return torch.mean(torch.abs(a[a>threshold] - b[a>threshold]))


# FINAL GRADIENT LOSS USED FOR EDGE LEARNING THROUGH DEPTHANYTHING
def dice_loss(gt_grad, out_grad, smooth=1e-3, threshold = 0.0):
    # Apply tanh
    out_grad = torch.tanh(torch.abs(out_grad))

    gt_grad = (gt_grad > threshold).float()
    #gt_grad = torch.sigmoid(gt_grad)
    
    # Calculate intersection and union
    intersection = (out_grad * gt_grad).sum(dim=(2, 3))
    union = (out_grad).sum(dim=(2, 3)) + (gt_grad).sum(dim=(2, 3))

    # Compute Dice Coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    
    # Return Dice Loss
    return 1 - dice.mean()


# RESCALES DEPTH IMAGE OBTAINED WITH DEPTH ANYTHING TO ROUGHLY THE SAME SCALE AS GT
def rescale_guidanceDA(guidance_DA, gt):
    x = guidance_DA/guidance_DA.max() 
    x_max = x.max()
    x_min = x.min()
    y_max = gt.max()
    y_min = gt[gt>0].min()

    y_new = ((x-x_min)/(x_max-x_min)) * (y_min-y_max) + y_max

    return y_new


# RETURNS RMSE WHEN DEPTH IMAGE WAS RESCALED USING MIN-MAX VALUES (ONLY IN NYU)
def calc_rmse_raw(gt, out, minmax):
    gt = gt[6:-6, 6:-6]
    out = out[6:-6, 6:-6]
    a = gt.clone()
    # rescales
    gt = gt*(minmax[1]-minmax[0]) + minmax[0]
    out = out*(minmax[1]-minmax[0]) + minmax[0]
    
    return torch.sqrt(torch.mean(torch.pow(gt[a>0]-out[a>0],2)))

# RETURNS RMSE WHEN DEPTH IMAGE WAS RESCALED USING A GLOBAL MAX VALUE
def calc_rmse_real(gt, out, max_value):
    a = gt.clone()
    # rescales
    gt = gt*max_value
    out = out*max_value
    
    return torch.sqrt(torch.mean(torch.pow(gt[a>0]-out[a>0],2)))


# SPLIT IMAGE IN TILES FOR FREQUENCY LOSS    
def split_in_tiles(img, split_h = 8, split_w = 8):
    h, w = img.shape[2], img.shape[3]
    #assert h%split_h==0 and w%split_w==0, 'height and width of your image should be multiple of your split_ratio'

    img = img.squeeze(0).squeeze(0)
    patches = img.unfold(0, h // split_h, h // split_h).unfold(1, w // split_w, w // split_w)
    patches = patches.reshape(-1, h // split_h, w // split_w)

    return patches

# DISCARD A PATCH IF ZERO COUNTS EXCEEDS THRESHOLD
def discard_bad_tiles(patches):
    zero_counts = (patches <= 0.05).sum(dim=(1, 2))
    mask = zero_counts < 10
    selected_channels = torch.nonzero(mask, as_tuple=True)[0]
    return patches[selected_channels], selected_channels

# DFT OF TILES
def fourier_transform_tiles(patches):
    net_getFrequency = getFrequency()
    amps = torch.empty(patches.shape, dtype=torch.float32)
    phases = torch.empty(patches.shape, dtype=torch.float32)
    for channel in range(patches.shape[0]):
        amps[channel], phases[channel] =  net_getFrequency(patches[channel])

    return amps, phases

# FREQUENCY LOSS
def custom_frequency_loss(out_fre, gt_fre):
    out_amps, out_pha = out_fre
    gt_amps, gt_pha = gt_fre
    loss = nn.L1Loss()
    channels = out_amps.shape[0]
    amp_loss = 0.0
    pha_loss = 0.0
    for idx in range(channels):
        amp_loss += loss(out_amps[idx], gt_amps[idx])
        pha_loss += loss(out_pha[idx], gt_pha[idx])

    return amp_loss/channels, pha_loss/channels
    

''' DEPRECATED STUFF


    #def custom_gradient_loss(gt_grad, out_grad, gt):
#    gt2 = gt.clone() 
#    gt2 = torch.from_numpy(erosion_augm(gt2[0,0].cpu().numpy(), kernel = 15)).unsqueeze(0).unsqueeze(0).cuda()

#    mask1 = (gt_grad > 0.0075) & (gt2 > 0)
#    loss1 = torch.mean(torch.abs(gt_grad[mask1] - out_grad[mask1])) * 100
   
#    mask2 = (gt_grad <= 0.0075) & (gt2 > 0)
#    loss2 = torch.mean(torch.abs(gt_grad[mask2] - out_grad[mask2])) 

#   loss = torch.nan_to_num(loss1) + torch.nan_to_num(loss2)

#   gt_grad[gt2 <= 0] = 0
#   return loss, gt_grad

#def gradient_loss_edges_DA_v1(a, b, threshold = 0.0):
#    a = (a > threshold).float() * 100.0
#    b = (b > threshold).float() * 100.0
#    
#    # MAXIMIZE INTERSECTION OVER UNION
#    return torch.mean(torch.abs(a - b))


#def ssim_structure_loss(img1, img2, window_size=11, C=1e-2):
#    """ Compute SSIM structure component between two images. """
#    # Create a Gaussian window for local statistics
#    device = img1.device
#    pad = window_size // 2
#    kernel = torch.ones(1, 1, window_size, window_size, device = device)/ (window_size ** 2)

#    mu1 = F.conv2d(img1, kernel, padding=pad)
#    mu2 = F.conv2d(img2, kernel, padding=pad)

#    # Compute variances & covariances
#    sigma1_sq = torch.clamp(F.conv2d(img1 * img1, kernel, padding=pad) - mu1**2, min=0.0)
#    sigma2_sq = torch.clamp(F.conv2d(img2 * img2, kernel, padding=pad) - mu2**2, min=0.0)
#    sigma12 = F.conv2d(img1 * img2, kernel, padding=pad) - mu1 * mu2

#    # Compute only the structure component of SSIM
#    structure = (sigma12 + C) / (torch.sqrt(sigma1_sq + 1e-6) * torch.sqrt(sigma2_sq + 1e-6) + C)
   
#    return 1 - structure.mean()  # Return mean structure similarity
 

#def gradient_loss_edges_GT_v1(gt_grad, out_grad, gt, threshold = 0.0):
#    # EROSION OF GT
#    gt2 = gt.clone() 
#    gt2 = torch.from_numpy(erosion_augm(gt2[0,0].cpu().numpy(), kernel = 15)).unsqueeze(0).unsqueeze(0).cuda()
    
#    mask = (gt2 <= 0)
#    gt_grad[mask] = 0
    
#    gt_grad = (gt_grad > threshold).float() * 100.0
#    out_grad = (out_grad > threshold).float() * 100.0
    
 #   # MAXIMIZE INTERSECTION OVER UNION
 #   return torch.mean(torch.abs(gt_grad - out_grad)), gt_grad


#def gradient_loss_GT_eroded(gt_grad, out_grad, gt):
#    # EROSION OF GT
#    gt2 = gt.clone() 
#    gt2 = torch.from_numpy(erosion_augm(gt2[0,0].cpu().numpy(), kernel = 15)).unsqueeze(0).unsqueeze(0).cuda()

    # MAE BETWEEN GT_GRAD AND OUT_GRAD ONLY FOR VALID VALUES OF GT AFTER EROSION
#    loss = torch.mean(torch.abs(gt_grad[gt2 > 0] - out_grad[gt2 > 0]))

#    mask = (gt2 <= 0)
#    mask[:, :, :3, :] = True
#    mask[:, :, -3:, :] = True
#    mask[:, :, :, :3] = True
#    mask[:, :, :, -3:] = True
#    gt_grad[mask] = 0

#    return loss, gt_grad
'''

    
