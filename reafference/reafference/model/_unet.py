#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
    CREDIT: https://github.com/milesial/Pytorch-UNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("UNet",)
class UNet(nn.Module):

    def __init__(self, n_channels, n_classes, condition=(lambda x, a: x), exp=6, output_activation=(lambda x : x), bilinear=False, batch_normalize=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.batch_normalize = batch_normalize
        
        factor = 2 if bilinear else 1
        assert exp >= 1

        self.inc = DoubleConv(n_channels, 2**exp,           batch_normalize=batch_normalize)
        self.down1 = Down(2**exp, 2**(exp+1),               batch_normalize=batch_normalize)
        self.down2 = Down(2**(exp+1), 2**(exp+2),           batch_normalize=batch_normalize)
        self.down3 = Down(2**(exp+2), 2**(exp+3),           batch_normalize=batch_normalize)
        self.down4 = Down(2**(exp+3), 2**(exp+4) // factor, batch_normalize=batch_normalize)
        self.up1 = Up(2**(exp+4), 2**(exp+3) // factor, bilinear, batch_normalize=batch_normalize)
        self.up2 = Up(2**(exp+3), 2**(exp+2) // factor, bilinear, batch_normalize=batch_normalize)
        self.up3 = Up(2**(exp+2), 2**(exp+1) // factor, bilinear, batch_normalize=batch_normalize)
        self.up4 = Up(2**(exp+1), 2**exp, bilinear, batch_normalize=batch_normalize)
        self.outc = OutConv(2**exp, n_classes)

        self.condition = condition 
        self.output_activation = output_activation

    def forward(self, x, a=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.condition(x5, a)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return self.output_activation(x)

    def conditional_shape(self, input_shape, device="cpu"):
        x1 = self.inc(torch.zeros((2, *input_shape), device=device))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return self.down4(x4).shape[1:]

class DoubleConv(nn.Module):
    """(convolution => [BN] => LeakyReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, batch_normalize=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels) if batch_normalize else nn.Identity(),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels) if batch_normalize else nn.Identity(),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, batch_normalize=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, batch_normalize=batch_normalize)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, batch_normalize=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, batch_normalize=batch_normalize)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, batch_normalize=batch_normalize)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)