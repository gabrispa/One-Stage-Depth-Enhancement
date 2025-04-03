import torch.nn as nn
import torch
from torchvision import transforms
import torchvision.transforms.functional as FT
import torch.nn.init as init
import torch.nn.functional as F
from models.common_modules import *
import matplotlib.pyplot as plt
from utils import *

class GetGradientRGB(nn.Module):
    def __init__(self):
        super(GetGradientRGB, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

        
    def forward(self, x):
        # input RGB image should be of shape = (1, 3, H, W) because we always use transform.toTensor 
        # after loading dataset in the main application

        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x

class GetGradientDepth(nn.Module):
    def __init__(self):
        super(GetGradientDepth, self).__init__()

        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        # input depth image should be of shape = (1, 1, H, W) because we always use transform.toTensor 
        # after loading dataset in the main application
      
        x0 = x[:,0]

        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)

        mask = torch.ones_like(x0)
        mask[:, :, :3, :] = 0
        mask[:, :, -3:, :] = 0
        mask[:, :, :, :3] = 0
        mask[:, :, :, -3:] = 0

        x = x0 * mask
        return x

## Channel Attention (CA) Layer
## wrt Squeeze-Excitement Networks a Conv2d is used instead of nn.Linear
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # feature channel downscale and upscale --> channel weight
        self.conv_down_up = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, kernel_size = 1, padding = 0, bias = True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, kernel_size = 1, padding = 0, bias = True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_down_up(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__( self, conv, n_feat, kernel_size, reduction,
                 bias = True, bn = False, act = nn.ReLU(True), res_scale = 1):
        super(RCAB, self).__init__()
        
        modules_body = []
        # convolution -> ReLU -> convolution  (optionally a batchNorm after conv)
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias = bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
                
        modules_body.append(CALayer(n_feat, reduction))
        
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(conv, n_feat, kernel_size, reduction, bias = True, bn = False, 
                act = nn.LeakyReLU(negative_slope = 0.2, inplace = True), res_scale = 1) \
            for _ in range(n_resblocks)]
        
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Block (no channel attention)
class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class GCM(nn.Module):
    def __init__(self, n_feats, scale):
        super(GCM, self).__init__()
        
        self.gradientRGB = GetGradientRGB()
        self.gradientDepth = GetGradientDepth()
        
        self.upBlock = DenseProjection(1, 1, scale, up=True, bottleneck=False)
        self.downBlock = DenseProjection(n_feats, n_feats, scale, up=False, bottleneck=False)
        
        self.conv1_rgb = default_conv(3, n_feats, 3)
        self.conv1_depth = default_conv(1, n_feats, 3)

        self.residualGroup_depth = ResidualGroup(default_conv, n_feats, 3, reduction = 16, n_resblocks = 4)
        
        self.resBlock_rgb = ResBlock(default_conv, n_feats, 3, bias = True, bn = False,
                                act = nn.LeakyReLU(negative_slope = 0.2, inplace = True), res_scale = 1)
        
        self.fuse_process = nn.Sequential(nn.Conv2d(2*n_feats, n_feats, kernel_size = 1),
                                          ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=4),
                                          ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=4))

        self.CA = CALayer(n_feats,reduction=4)
        self.conv_fused = default_conv(n_feats, n_feats, 3)
        
        self.refine_grad = default_conv(n_feats,1,3)
        
        self.refine_depth = default_conv(n_feats,1,3)
        
        #self.c_sab = default_conv(1,n_feats,3)
        #self.sig = nn.Sigmoid()
        
        self.depth_resG = nn.Sequential(default_conv(1,n_feats,3),
                                ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=8))


        grad_conv = [
            default_conv(1, n_feats, kernel_size=3, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            default_conv(n_feats, n_feats, kernel_size=3, bias=True),
        ]
        self.grad_conv = nn.Sequential(*grad_conv)
        
        self.final_resG = nn.Sequential(ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=4),
                                    ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=4))

    def forward(self, depth, rgb):
        depth = self.upBlock(depth)
    
        gradientRGB = self.gradientRGB(rgb)
        gradientDepth = self.gradientDepth(depth)

        d1 = self.conv1_depth(gradientDepth)
      
        d2 = self.residualGroup_depth(d1) # depth feature map processed through the residual group

        rgb1 = self.conv1_rgb(gradientRGB)
        rgb2 = self.resBlock_rgb(rgb1)  # rgb feature map processed through the residual block

        cat1 = torch.cat([rgb2, FT.resize(d2, (rgb2.shape[2], rgb2.shape[3]))], dim = 1)  # concatenation of the two previously obtained feature maps

        inn1 = self.fuse_process(cat1)

        d3 = d1 + self.CA(inn1) # combines the fused features (passed through channel attention) 
                                # with the initial depth gradient features
        grad_d2 = self.conv_fused(d3)

        gradient_SR = self.refine_grad(grad_d2)  # Gradient SR 

        d4 = self.depth_resG(depth)

        grad_d3 = self.grad_conv(gradient_SR) + d4

        grad_d4 = self.final_resG(grad_d3)
    
        return gradient_SR, self.downBlock(grad_d4)