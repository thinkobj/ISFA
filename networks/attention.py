
import numpy as np
import torch
import math
import torch.nn as nn
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module', 'CAM_Module', 'SELayer']


class PAM_Module(Module):
    """ Edge attention module in Supervised Edge Attention Network for Accurate Image Instance Segmentation"""

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class EAM_Module(Module):
    """ Edge attention module in Supervised Edge Attention Network for Accurate Image Instance Segmentation"""

    def __init__(self, in_dim):
        super(EAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        out = self.conv(out)
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)  # [1, 256, 256]

        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)  # [1, 196, 256]

        energy = torch.bmm(proj_query, proj_key)  # [1, 256, 256]

        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy  # [1, 256, 256]
        attention = self.softmax(energy_new)  # [1, 256, 256]

        proj_value = x.view(m_batchsize, C, -1)  # [1, 256, 196]

        out = torch.bmm(attention, proj_value)   # [1, 256, 196]

        out = out.view(m_batchsize, C, height, width)  # [1, 256, 14, 14]
        out = self.gamma*out + x
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=5, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=5, dilation=1),
        )
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.conv(x)
        b, c, _, _ = x1.size()
        y = self.avg_pool(x1).view(b, c)  # [1, 256]

        y = self.fc(y).view(b, c, 1, 1)  # [1, 256, 1, 1]
        return x * y.expand_as(x)


if __name__ == '__main__':
    input = torch.ones(1, 256, 14, 14)
    pam = SELayer(channel=256)
    out = pam(input)
    print(out.shape)
