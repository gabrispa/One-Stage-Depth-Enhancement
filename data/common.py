import random
import numpy as np
import cv2 as cv

def augment(imgs, hflip=True, vflip=True):
    hflip = hflip and random.random() < 0.5
    vflip = vflip and random.random() < 0.5

    augmented_imgs = []
    
    for img in imgs:
        # Apply horizontal flip
        if hflip:
            img = img[:, ::-1, :].copy()
        
        # Apply vertical flip
        if vflip:
            img = img[::-1, :, :].copy()
        
        augmented_imgs.append(img)
    return augmented_imgs

def get_patch(imgs, patch_size=16):
    th, tw = imgs[0].shape[0], imgs[0].shape[1]

    tp = round(patch_size)
    tx = random.randrange(0, (tw-tp))
    ty = random.randrange(0, (th-tp))
    patches = []

    for img in imgs:
        img = img[ty:ty + tp, tx:tx + tp, :]
        patches.append(img)

    return patches

def get_patch_metagrasp(imgs, patch_size=16):
    th, tw = imgs[0].shape[0], imgs[0].shape[1]

    tp = round(patch_size)
    tx = random.randrange(200, (tw-tp-200))
    ty = random.randrange(100, (th-tp-100))
    patches = []

    for img in imgs:
        img = img[ty:ty + tp, tx:tx + tp, :]
        patches.append(img)

    return patches

# DEPRECATED SINCE THE TRAINING IMAGE IS SPLIT IN PATCHES FED SEQUENTIALLY INTO THE NETWORK
def get_patch_realdata(imgs, lr, patch_size=16, scale=2):
    th, tw = imgs[0].shape[0], imgs[0].shape[1]

    tp = round(patch_size)
    tx = 1100# random.randrange(300, (tw-tp-500),2)
    ty = 1100# random.randrange(50, (th-tp-100),2)
    patches = []

    for img in imgs:
        img = img[ty:ty + tp, tx:tx + tp, :]
        patches.append(img)

    patch_lr = lr[ty//scale:(ty+tp)//scale, tx//scale:(tx+tp)//scale, :]
    patches.append(patch_lr)

    return patches

def erosion_augm(lr, kernel):
    x = lr
    mask = (x<1e-5).astype(np.float32)
    mask = cv.dilate(mask, np.ones((kernel,kernel), np.uint8), iterations=1).astype(bool)
    x[mask] = 0
    return x

def add_zero_pixels(depth, noise_ratio = 0.0):
    h, w = depth.shape[0], depth.shape[1]
        
    number_of_pixels = int(h*w*noise_ratio)

    for i in range(number_of_pixels):
        y_coord = random.randint(0,h-1)
        x_coord = random.randint(0,w-1)
        
        depth[y_coord][x_coord] = 0.0  
    return depth
    

def add_salt_pepper(depth, mode):
    h, w = depth.shape[0], depth.shape[1]
    if mode:
        noise_ratio = 0.015
    else:
        noise_ratio = 0.01
        
    number_of_pixels = int(h*w*noise_ratio)

    for i in range(number_of_pixels):
        y_coord_salt = random.randint(0,h-1)
        y_coord_pepper = random.randint(0,h-1)
        x_coord_salt = random.randint(0,w-1)
        x_coord_pepper = random.randint(0,w-1)

        depth[y_coord_salt][x_coord_salt] = 1.0
        depth[y_coord_pepper][x_coord_pepper] = 0.0
    
    return depth

def add_gaussian_noise(depth, mean, stdv):
    gaussian_noise = np.random.normal(mean, stdv, depth.shape)
    return depth + gaussian_noise

def add_big_noise(depth, noise_ratio):
    h, w = depth.shape[0], depth.shape[1]
    number_of_pixels = int(h*w*noise_ratio)

    for i in range(number_of_pixels):
        y_coord = random.randint(0,h-1)
        x_coord = random.randint(0,w-1)
        depth[y_coord][x_coord] = 0.0
    return depth
