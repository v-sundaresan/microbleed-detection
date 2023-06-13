from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from microbleednet.utils import microbleednet_model_utils
import numpy as np
import torch
import torch.nn.functional as F

#=========================================================================================
# MicrobleedNet Candidate Detection and Discrimination model
# Vaanathi Sundaresan
# 09-01-2023
#=========================================================================================


class CDetNet(nn.Module):
    """
    Microbleednet Candidate Detection Model definition
    """
    def __init__(self, n_channels, n_classes, init_channels, bilinear=False):
        super(CDetNet, self).__init__()
        self.n_channels = n_channels
        self.init_channels = init_channels
        self.n_classes = n_classes
        self.n_layers = 3
        self.bilinear = bilinear

        self.inpconv = microbleednet_model_utils.OutConv(n_channels, 3)
        self.convfirst = microbleednet_model_utils.DoubleConv(3, init_channels, 3, 1)
        self.down1 = microbleednet_model_utils.Down(init_channels, init_channels*2, 3, 1)
        self.down2 = microbleednet_model_utils.Down(init_channels*2, init_channels*4, 3, 1)
        self.up2 = microbleednet_model_utils.Up(init_channels*4, init_channels*2, 3, bilinear)
        self.up1 = microbleednet_model_utils.Up(init_channels*2, init_channels, 3, bilinear)
        self.outconv = microbleednet_model_utils.OutConv(init_channels, n_classes)

    def forward(self, x):
        xi = self.inpconv(x)
        x1 = self.convfirst(xi)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up2(x3, x2)
        x = self.up1(x, x1)
        logits = self.outconv(x)
        return logits


class CDiscNet(nn.Module):
    """
    Microbleednet Candidate Detection Model definition
    """
    def __init__(self, n_channels, n_classes, init_channels, bilinear=False):
        super(CDiscNet, self).__init__()
        self.n_channels = n_channels
        self.init_channels = init_channels
        self.n_classes = n_classes
        self.n_layers = 3
        self.bilinear = bilinear

        self.inpconv = microbleednet_model_utils.OutConv(n_channels, 3)
        self.convfirst = microbleednet_model_utils.DoubleConv(3, init_channels, 3, 1)
        self.down1 = microbleednet_model_utils.Down(init_channels, init_channels*2, 3, 1)
        self.down2 = microbleednet_model_utils.Down(init_channels*2, init_channels*4, 3, 1)
        self.up2 = microbleednet_model_utils.Up(init_channels*4, init_channels*2, 3, bilinear)
        self.up1 = microbleednet_model_utils.Up(init_channels*2, init_channels, 3, bilinear)
        self.outconv = microbleednet_model_utils.OutConv(init_channels, n_classes)

    def forward(self, x):
        xi = self.inpconv(x)
        x1 = self.convfirst(xi)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up2(x3, x2)
        x = self.up1(x, x1)
        logits = self.outconv(x)
        return x3, logits


class CDiscClass24(nn.Module):
    """
        Microbleednet Candidate Discrimination Model definition
    """
    def __init__(self, n_channels, n_classes, init_channels, bilinear=False):
        super(CDiscClass24, self).__init__()
        self.n_channels = n_channels
        self.init_channels = init_channels
        self.n_classes = n_classes
        self.n_layers = 3
        self.bilinear = bilinear

        self.convfirst = microbleednet_model_utils.SingleConv(init_channels, init_channels//2, 1)
        self.down1 = microbleednet_model_utils.Down(init_channels//2, init_channels//2, 3, 3)
        self.down2 = microbleednet_model_utils.Down(init_channels//2, init_channels//2, 3, 3)
        self.fc1 = nn.Linear(512*2, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        xi = self.convfirst(x)
        x1 = self.down1(xi)
        x1 = self.down2(x1)
        x1 = x1.view(-1, 512*2)
        x1 = self.fc1(x1)
        x1 = self.fc2(x1)
        x1 = self.fc3(x1)
        return torch.sigmoid(x1)


class CDiscClass32(nn.Module):
    def __init__(self, n_channels, n_classes, init_channels, bilinear=False):
        super(CDiscClass32, self).__init__()
        self.n_channels = n_channels
        self.init_channels = init_channels
        self.n_classes = n_classes
        self.n_layers = 3
        self.bilinear = bilinear

        self.convfirst = microbleednet_model_utils.SingleConv(init_channels, init_channels//2, 1)
        self.down1 = microbleednet_model_utils.Down(init_channels, init_channels, 3, 3)
        self.down2 = microbleednet_model_utils.Down(init_channels, init_channels, 3, 3)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        xi = self.convfirst(x)
        x1 = self.down1(xi)
        x1 = self.down2(x1)
        x1 = x1.view(-1, 512)
        x1 = self.fc1(x1)
        x1 = self.fc2(x1)
        x1 = self.fc3(x1)
        return torch.sigmoid(x1)


class CDiscClass48(nn.Module):
    def __init__(self, n_channels, n_classes, init_channels, bilinear=False):
        super(CDiscClass48, self).__init__()
        self.n_channels = n_channels
        self.init_channels = init_channels
        self.n_classes = n_classes
        self.n_layers = 3
        self.bilinear = bilinear

        self.convfirst = microbleednet_model_utils.SingleConv(init_channels, init_channels//2, 1)
        self.down1 = microbleednet_model_utils.Down(init_channels, init_channels, 3, 3)
        self.down2 = microbleednet_model_utils.Down(init_channels, init_channels, 3, 3)
        self.down3 = microbleednet_model_utils.Down(init_channels, init_channels, 3, 1)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        xi = self.convfirst(x)
        x1 = self.down1(xi)
        x1 = self.down2(x1)
        x1 = self.down3(x1)
        x1 = x1.view(-1, 512)
        x1 = self.fc1(x1)
        x1 = self.fc2(x1)
        x1 = self.fc3(x1)
        return torch.sigmoid(x1)


class CDiscStudentNet(nn.Module):
    def __init__(self, n_channels, n_classes, init_channels, bilinear=False):
        super(CDiscStudentNet, self).__init__()
        self.n_channels = n_channels
        self.init_channels = init_channels
        self.n_classes = n_classes
        self.n_layers = 3
        self.bilinear = bilinear

        self.inpconv = microbleednet_model_utils.OutConv(n_channels, 3)
        self.convfirst = microbleednet_model_utils.DoubleConv(3, init_channels, 3, 1)
        self.down1 = microbleednet_model_utils.Down(init_channels, init_channels * 2, 3, 1)
        self.down2 = microbleednet_model_utils.Down(init_channels * 2, init_channels * 4, 3, 1)

        self.classconvfirst = microbleednet_model_utils.SingleConv(init_channels * 4, init_channels * 2, 1)
        self.classdown1 = microbleednet_model_utils.Down(init_channels * 2, init_channels * 2, 3, 3)
        self.classdown2 = microbleednet_model_utils.Down(init_channels * 2, init_channels * 2, 3, 3)
        # self.down3 = Down(init_channels, init_channels//2, 1)
        self.fc1 = nn.Linear(512 * 2, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        xi = self.inpconv(x)
        x1 = self.convfirst(xi)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        xi = self.classconvfirst(x3)
        x1 = self.classdown1(xi)
        x1 = self.classdown2(x1)
        x1 = x1.view(-1, 512 * 2)
        x1 = self.fc1(x1)
        x1 = self.fc2(x1)
        x1 = self.fc3(x1)
        return torch.sigmoid(x1)

