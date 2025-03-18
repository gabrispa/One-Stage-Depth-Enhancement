import torch.nn as nn
import torch
import torch.nn.init as init
import torch.nn.functional as F
from models.common_modules import *

class FrequencyDiff(nn.Module):
    def __init__(self, depth_channels, rgb_channels):
        super(FrequencyDiff, self).__init__()
        channels = depth_channels
        
        self.pre_rgb = unitary_kernel_conv(rgb_channels, channels)
        self.pre_depth = unitary_kernel_conv(channels, channels)
        
        self.fuse_conv = nn.Sequential(unitary_kernel_conv(channels, channels),
                                    nn.LeakyReLU(negative_slope = 0.1, inplace=False),
                                    unitary_kernel_conv(channels, channels))
        self.fuse_sub = nn.Sequential(unitary_kernel_conv(channels, channels),
                                      nn.LeakyReLU(negative_slope = 0.1, inplace=False),
                                      unitary_kernel_conv(channels, channels))
        self.post = unitary_kernel_conv(2*channels, channels) 
        #self.sig = nn.Sigmoid()

    def forward(self, dp, rgb):

        dp1 = self.pre_depth(dp)
        rgb1 = self.pre_rgb(rgb)

        fuse_c = self.fuse_conv(dp1)

        fuse_sub = self.fuse_sub(torch.abs(rgb1 - dp1))
        cat_fuse = torch.cat([fuse_c, fuse_sub], dim = 1)

        return self.post(cat_fuse)

class SubSDB(nn.Module):
    def __init__(self, depth_channels, rgb_channels):
        super(SubSDB, self).__init__()
        channels = depth_channels
        
        self.pre1 = unitary_kernel_conv(channels, channels)
        self.pre2 = unitary_kernel_conv(rgb_channels, rgb_channels)
        
        self.amp_fuse = FrequencyDiff(channels, rgb_channels)
        self.pha_fuse = FrequencyDiff(channels, rgb_channels)
        
        self.post = unitary_kernel_conv(channels, channels)

    def forward(self, dp, rgb):
        _, _, H, W = dp.shape
        dp = torch.fft.rfft2(self.pre1(dp) + 1e-8, norm = 'backward')
        rgb = torch.fft.rfft2(self.pre2(rgb) + 1e-8, norm = 'backward')
        
        dp_amp = torch.abs(dp)
        dp_pha = torch.angle(dp)
        rgb_amp = torch.abs(rgb)
        rgb_pha = torch.angle(rgb)
        amp_fuse = self.amp_fuse(dp_amp, rgb_amp)
        pha_fuse = self.pha_fuse(dp_pha, rgb_pha)

        real = amp_fuse * torch.cos(pha_fuse)+1e-8
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8
        out = torch.complex(real, imag)+1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))

        return self.post(out)

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, d = 1, init='xavier', gc=8, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc, d)
        self.conv2 = UNetConvBlock(gc, gc, d)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3

class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not LU_decomposed:
            # Sample a random orthogonal matrix:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape
        self.LU = LU_decomposed

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        if not self.LU:
            pixels = thops.pixels(input)
            dlogdet = torch.slogdet(self.weight)[1] * pixels
            if not reverse:
                weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
            else:
                weight = torch.inverse(self.weight.double()).float()\
                              .view(w_shape[0], w_shape[1], 1, 1)
            return weight, dlogdet
        else:
            self.p = self.p.to(input.device)
            self.sign_s = self.sign_s.to(input.device)
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)
            l = self.l * self.l_mask + self.eye
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            dlogdet = thops.sum(self.log_s) * thops.pixels(input)
            if not reverse:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:
                l = torch.inverse(l.double()).float()
                u = torch.inverse(u.double()).float()
                w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
            return w.view(w_shape[0], w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)
        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet

class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, d = 1, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1, d)
        self.G = subnet_constructor(self.split_len1, self.split_len2, d)
        self.H = subnet_constructor(self.split_len1, self.split_len2, d)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out

class SDB(nn.Module):
    def __init__(self, channels, rgb_channels, scale):
        super(SDB, self).__init__()
        
        self.rgbprocess = default_conv(rgb_channels, rgb_channels, 3)
        self.rgbpre =  unitary_kernel_conv(rgb_channels, rgb_channels)
        
        self.spa_process = nn.Sequential(InvBlock(DenseBlock, channels + rgb_channels, channels),
                                         nn.Conv2d(channels + rgb_channels, channels, 1, 1, 0))
        self.fre_process = SubSDB(channels, rgb_channels)
        self.spa_att = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1, bias=True),
                                     nn.Sigmoid())
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels
        self.cha_att = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=1, bias=True),
                                     nn.LeakyReLU(negative_slope = 0.1),
                                     nn.Conv2d(channels // 2, channels, kernel_size=1, bias=True),
                                     nn.Sigmoid())
        self.post = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)

        self.fuse_process = nn.Sequential(InvBlock(DenseBlock, 2*channels, channels),
                                         nn.Conv2d(2*channels, channels, kernel_size = 1))

        self.downBlock = DenseProjection(channels, channels, scale, up=False, bottleneck=False)
        self.upBlock = DenseProjection(channels, channels, scale, up=True, bottleneck=False)

    def forward(self, dp, rgb): 
        dp = self.upBlock(dp)

        rgbpre = self.rgbprocess(rgb)
        rgb = self.rgbpre(rgbpre)

        spafuse = self.spa_process(torch.cat([dp, rgb], dim = 1))
    
        frefuse = self.fre_process(dp, rgb)

        cat_f = torch.cat([spafuse, frefuse], dim = 1)
        cat_f = self.fuse_process(cat_f)

        cha_res = self.cha_att(self.contrast(cat_f) + self.avgpool(cat_f)) * cat_f
        out = cha_res + dp

        out = self.downBlock(out)

        return out, rgbpre