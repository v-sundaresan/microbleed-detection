#!/usr/bin/env python
#   Copyright (C) 2016 University of Oxford
#   SHBASECOPYRIGHT

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import *


class MedStudentNet(nn.Module):
    def __init__(self, n_channels, n_classes, batch_size, init_channels, bilinear=False):
        super(MedStudentNet, self).__init__()
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.init_channels = init_channels
        self.n_classes = n_classes
        self.n_layers = 3
        self.bilinear = bilinear

        self.inpconv = OutConv(n_channels, 3)
        self.convfirst = DoubleConv(3, init_channels, 3, 1)
        self.down1 = Down(init_channels, init_channels * 2, 3, 1)
        self.down2 = Down(init_channels * 2, init_channels * 4, 3, 1)

        self.classconvfirst = SingleConv(init_channels * 4, init_channels * 2, 1)
        self.classdown1 = Down(init_channels * 2, init_channels * 2, 3, 3)
        self.classdown2 = Down(init_channels * 2, init_channels * 2, 3, 3)
        # self.down3 = Down(init_channels, init_channels//2, 1)
        self.fc1 = nn.Linear(512 * 2, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        xi = self.inpconv(x)
        print(xi.size())
        x1 = self.convfirst(xi)
        print(x1.size())
        x2 = self.down1(x1)
        print(x2.size())
        x3 = self.down2(x2)
        print(x3.size())
        xi = self.classconvfirst(x3)
        print(xi.size())
        x1 = self.classdown1(xi)
        print(x1.size())
        x1 = self.classdown2(x1)
        print(x1.size())
        x1 = x1.view(-1, 512 * 2)
        print(x1.size())
        x1 = self.fc1(x1)
        print(x1.size())
        x1 = self.fc2(x1)
        print(x1.size())
        x1 = self.fc3(x1)
        print(x1.size())
        return torch.sigmoid(x1)

