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

class MedNet(nn.Module):
    def __init__(self, n_channels, n_classes, batch_size, init_channels, bilinear=False):
        super(MedNet, self).__init__()
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.init_channels = init_channels
        self.n_classes = n_classes
        self.n_layers = 3
        self.bilinear = bilinear

        self.inpconv = OutConv(n_channels, 3)
        self.convfirst = DoubleConv(3, init_channels, 3, 1)
        self.down1 = Down(init_channels, init_channels*2, 3, 1)
        self.down2 = Down(init_channels*2, init_channels*4, 3, 1)
        #self.down3 = Down(init_channels*4, init_channels*8, 3)
        #factor = 2 if bilinear else 1
        #self.up3 = Up(init_channels*8, init_channels*4, 3, bilinear)
        self.up2 = Up(init_channels*4, init_channels*2, 3, bilinear)
        self.up1 = Up(init_channels*2, init_channels, 3, bilinear)
        self.outconv = OutConv(init_channels, n_classes)

    def forward(self, x):
        xi = self.inpconv(x)
        #print(xi.size())
        x1 = self.convfirst(xi)
        #print(x1.size())
        x2 = self.down1(x1)
        #print(x2.size())
        x3 = self.down2(x2)
        #print(x3.size())
        #x4 = self.down3(x3)
        #x = self.up3(x4, x3)
        x = self.up2(x3, x2)
        x = self.up1(x, x1)
        logits = self.outconv(x)
        return logits
