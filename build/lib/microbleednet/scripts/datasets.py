from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import random
import numpy as np
from scipy.ndimage import filters
from torch.utils.data import Dataset

from microbleednet.scripts import data_preparation

#####################################
# Truenet dataset utility functions #
# Vaanathi Sundaresan               #
# 09-03-2021, Oxford                #
#####################################


class CDiscPatchDataset(Dataset):

    def __init__(self, minority_class, majority_class, ratio, perform_augmentations=False):
        self.minority_class = minority_class
        self.majority_class = majority_class

        self.minority_class_copy = list(minority_class)
        self.majority_class_copy = list(majority_class)

        self.ratio = ratio
        self.perform_augmentations = perform_augmentations

        if self.ratio == 'random':
            self.full_set = self.minority_class + self.majority_class

    def reset_samples(self):
        if len(self.majority_class) < 2:
            self.majority_class = list(self.majority_class_copy)
        if len(self.minority_class) < 2:
            self.minority_class = list(self.minority_class_copy)
    
    def process(self, data):

        x = data['data_patch']
        y = data['patch_label']

        x = np.expand_dims(x, 0)
        y = np.expand_dims(y, 0)

        if self.perform_augmentations:
            x, _= data_preparation.augment_data(x, np.zeros_like(x), n_augmentations=4)
            y = np.stack([y] * 5, axis=0)
            y = y[:, 0]

        return x, y

    def sample(self, index):

        if self.ratio == '1:1':
            x1 = self.minority_class[index]

            if len(self.majority_class) < 1:
                self.reset_samples()

            # Take majority_class samples without replacement
            index = random.randint(0, len(self.majority_class) - 1)
            x2 = self.majority_class[index]
            self.majority_class.pop(index)

            samples = [x1, x2]
        
        elif self.ratio == '2:1':

            if len(self.majority_class) < 1:
                self.reset_samples()

            index = random.randint(0, len(self.majority_class) - 1)
            x1 = self.majority_class[index]
            self.majority_class.pop(index)

            if len(self.minority_class) < 2:
                self.reset_samples()

            # Take minority_class samples without replacement
            index = random.randint(0, len(self.minority_class) - 1)
            x2 = self.minority_class[index]
            self.minority_class.pop(index)

            index = random.randint(0, len(self.minority_class) - 1)
            x3 = self.minority_class[index]
            self.minority_class.pop(index)

            samples = [x1, x2, x3]

        elif self.ratio == 'random':
            x = self.full_set[index]
            samples = [x]

        return samples

    def __getitem__(self, index):

        samples = self.sample(index)

        x_store, y_store = zip(*[self.process(sample) for sample in samples])

        x = np.concatenate(x_store, axis=0)
        y = np.concatenate(y_store, axis=0)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        data_dict = {
            'input': x,
            'label': y,
        }

        return data_dict

    def __len__(self):
        if self.ratio == '1:1':
            return len(self.minority_class)
        elif self.ratio == '2:1':
            return len(self.minority_class)
        elif self.ratio == 'random':
            return len(self.full_set)
        

class CDetPatchDataset(Dataset):

    def __init__(self, minority_class, majority_class, ratio, perform_augmentations=False):

        self.minority_class = minority_class
        self.majority_class = majority_class

        self.minority_class_copy = list(minority_class)
        self.majority_class_copy = list(majority_class)

        self.ratio = ratio
        self.perform_augmentations = perform_augmentations

    def reset_samples(self):
        if len(self.majority_class) < 2:
            self.majority_class = list(self.majority_class_copy)
        if len(self.minority_class) < 2:
            self.minority_class = list(self.minority_class_copy)
    
    def process(self, data):

        x = data['data_patch']
        y = data['label_patch']
        pixel_weights = data['patch_pixel_weights']

        x = np.expand_dims(x, 0)
        y = np.expand_dims(y, 0)
        pixel_weights = np.expand_dims(pixel_weights, 0)

        if self.perform_augmentations:
            x, y = data_preparation.augment_data(x, y, n_augmentations=4)
            pixel_weights = filters.gaussian_filter(y, 1.2) * 10
        
        y = np.stack((1 - y, y), axis=1)
        pixel_weights = np.expand_dims(pixel_weights, 1)

        return x, y, pixel_weights
    
    def sample(self, index):

        if self.ratio == '1:1':
            x1 = self.minority_class[index]

            # Take majority_class samples without replacement
            index = random.randint(0, len(self.majority_class) - 1)
            x2 = self.majority_class[index]
            self.majority_class.pop(index)

            samples = [x1, x2]
        
        elif self.ratio == '2:1':

            if len(self.majority_class) < 1:
                self.reset_samples()

            index = random.randint(0, len(self.majority_class) - 1)
            x1 = self.majority_class[index]
            self.majority_class.pop(index)

            if len(self.minority_class) < 2:
                self.reset_samples()

            # Take minority_class samples without replacement
            index = random.randint(0, len(self.minority_class) - 1)
            x2 = self.minority_class[index]
            self.minority_class.pop(index)

            index = random.randint(0, len(self.minority_class) - 1)
            x3 = self.minority_class[index]
            self.minority_class.pop(index)

            samples = [x1, x2, x3]

        return samples

    def __getitem__(self, index):

        samples = self.sample(index)

        x_store, y_store, pixel_weights_store = zip(*[self.process(sample) for sample in samples])

        x = np.concatenate((x_store), axis=0)
        y = np.concatenate((y_store), axis=0)
        pixel_weights = np.concatenate((pixel_weights_store), axis=0)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        pixel_weights = torch.from_numpy(pixel_weights)

        data_dict = {
            'input': x,
            'label': y,
            'pixel_weights': pixel_weights
        }

        return data_dict

    def __len__(self):
        if self.ratio == '1:1':
            return len(self.minority_class)
        elif self.ratio == '2:1':
            return len(self.majority_class)

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



