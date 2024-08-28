import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
#from another_deform_conv import DeformConv2D
#from deform_conv_v2 import DeformConv2d
from SDUNet_SConv import SpatialDeformConv as SDConv

""" SDCN Block """


class conv_block(nn.Module):
    """ Standard Convolution Block """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class SDCN_block(nn.Module):
    """ parallel connection & pointwise convolution & DropBlock """
    
    def __init__(self, in_ch:int = 1, out_ch:int = 2, keep_prob:float = 0.2, drop_size = 3):
        super().__init__()
        self.deform_conv = nn.Sequential(
            SDConv(in_ch, out_ch, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True),
            SDConv(out_ch, out_ch, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True)
        )
        
        # self.deform_conv = SDConv(in_ch, out_ch, kernel_size = 3, padding = 1),
        # self.bn = nn.BatchNorm2d(out_ch),
        # self.relu = nn.ReLU(inplace = True),
        # self.deform_conv = SDConv(out_ch, out_ch, kernel_size = 3, padding = 1),
        # self.bn = nn.BatchNorm2d(out_ch),
        # self.relu = nn.ReLU(inplace = True)

        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True),
        )

        self.pconv = nn.Conv2d(2*out_ch, out_ch, kernel_size = 1, padding = "same", stride = 1)
        #self.dropblock = DropBlock2D(keep_prob, drop_size)
    
    def forward(self, x):
        x1 = self.deform_conv(x)
        x2 = self.conv(x)
        x = torch.cat([x1, x2], dim = 1) 
        x = self.pconv(x)
        #x = self.dropblock(x)
        return x


""" DropBlock """
""" https://github.com/alessandrolamberti/DropBlock """


class DropBlock2D(nn.Module):
    def __init__(self, p: float = 0.5, block_size: int = 3):
        self.block_size = block_size
        self.p = p

    def calculate_gamma(self, x: Tensor) -> float:
        """Computes gamma, eq. 1 in the paper
        args:
            x (Tensor): Input tensor
        returns:
            float: gamma
        """
        
        to_drop = (1 - self.p) / (self.block_size ** 2)
        to_keep = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        return to_drop * to_keep

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            gamma = self.calculate_gamma(x)
            mask = torch.bernoulli(torch.ones_like(x) * gamma)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )
            x = mask_block * x * (mask_block.numel() / mask_block.sum()) # normalize
        return x