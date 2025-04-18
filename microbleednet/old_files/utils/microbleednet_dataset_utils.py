from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.utils.data import Dataset

#=========================================================================================
# Truenet dataset utility functions
# Vaanathi Sundaresan
# 09-03-2021, Oxford
#=========================================================================================


class CMBTestDataset(Dataset):
    """This is a generic class for 2D segmentation datasets.
    :param data: stack of 3D slices N x C x H x W
    :param transform: transformations to apply.
    """
    def __init__(self, data, transform=None):
        self.data = torch.from_numpy(data).float()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]

        data_dict = {
            'input': x
        }

        if self.transform:
            data_dict = self.transform(data_dict)

        return data_dict

    def __len__(self):
        return len(self.data)


class CMBDataset(Dataset):
    """This is a generic class for 2D segmentation datasets.
    :param data: stack of 3D slices N x C x H x W
    :param target: stack of 3D slices N x C x H x W
    :param transform: transformations to apply.
    """
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).float()
        self.transform = transform # This is where you can add augmentations

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        data_dict = {
            'input': x,
            'gt': y
        }

        if self.transform:
            data_dict = self.transform(data_dict)

        return data_dict

    def __len__(self):
        return len(self.data)


class CMBDatasetWeighted(Dataset):
    """This is a generic class for 2D segmentation datasets.
    :param data: stack of 3D slices N x C x H x W
    :param label: stack of 3D slices N x C x H x W
    :param pixweights: stack of 2D slices N x C x H x W
    :param transform: transformations to apply.
    """
    def __init__(self, data, target, pixweights, transform=None):
        self.transform = transform
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).float()
        self.pixweights = torch.from_numpy(pixweights).float()

    def __getitem__(self, index):
        x = self.data[index]
        pw = self.pixweights[index]
        y = self.target[index]

        data_dict = {
            'input': x,
            'pixweights': pw,
            'gt': y
        }

        if self.transform:
            data_dict = self.transform(data_dict)

        return data_dict

    def __len__(self):
        return len(self.data)



