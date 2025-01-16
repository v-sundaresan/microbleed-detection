from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import random
import numpy as np
from collections import OrderedDict

###########################################
# Microbleednet general utility functions #
# Vaanathi Sundaresan                     #
# 09-01-2023                              #
###########################################

def create_patch_store(patches_store, indices):
    return [patches_store[idx] for idx in indices]

def split_patches(positive_patches_store, negative_patches_store, train_proportion):

    """
    Randomly split patches into training and validation sets
    :param positive_patches:
    :param negative_patches:
    :param train_proportion: training_params['Train_prop']
    :return: tuple containing the training paths, validation paths, and validation indices
    """

    n_positive_patches = len(positive_patches_store)
    n_negative_patches = len(negative_patches_store)

    n_patches = min(n_positive_patches, n_negative_patches)
    n_validation_patches = max(int(n_patches * (1 - train_proportion)), 1)

    positive_validation_indices = random.choices(list(np.arange(n_patches)), k=n_validation_patches)
    positive_train_indices = np.setdiff1d(np.arange(n_positive_patches), positive_validation_indices)

    negative_validation_indices = random.choices(list(np.arange(n_patches)), k=n_validation_patches)
    negative_train_indices = np.setdiff1d(np.arange(n_negative_patches), negative_validation_indices)

    train_patches_store = {
        'positive': create_patch_store(positive_patches_store, positive_train_indices),
        'negative': create_patch_store(negative_patches_store, negative_train_indices),
    }

    validation_patches_store = {
        'positive': create_patch_store(positive_patches_store, positive_validation_indices),
        'negative': create_patch_store(negative_patches_store, negative_validation_indices),
    }
    
    return train_patches_store, validation_patches_store

def split_subjects(subjects, train_proportion):
    
    """
    Randomly split data subjects into training and validation sets.
    :param data_paths: list of file paths to be split
    :param num_validation_subjects: int, number of validation subjects
    :return: tuple containing the training paths, validation paths, and validation indices
    """

    n_validation_subjects = max(int(len(subjects) * (1 - train_proportion)), 1)

    validation_indices = random.choices(list(np.arange(len(subjects))), k=n_validation_subjects)
    train_indices = np.setdiff1d(np.arange(len(subjects)), validation_indices)

    train_subjects = [subjects[idx] for idx in train_indices]
    validation_subjects = [subjects[idx] for idx in validation_indices]

    return train_subjects, validation_subjects, validation_indices

def freeze_layers_for_finetuning(model, layers_to_finetune, verbose=False):
    """
    Unfreezing specific layers of the model for fine-tuning
    :param model: model
    :param layer_to_ft: list of ints, layers to fine-tune starting from the decoder end.
    :param verbose: bool, display debug messages
    :return: model after unfreezing only the required layers
    """

    model_layer_names = ['outconv', 'up1', 'up2', 'up3', 'down3', 'down2', 'down1', 'convfirst']

    model_layers_to_finetune = [model_layer_names[layer_idx - 1] for layer_idx in layers_to_finetune]

    for name, child in model.module.named_children():
        if name in model_layers_to_finetune:
            if verbose:
                print(f'Model parameters in {name} are unfrozen')
            for param in child.parameters():
                param.requires_grad = True
        else:
            if verbose:
                print(f'Model parameters in {name} are frozen')
            for param in child.parameters():
                param.requires_grad = False

    return model

def load_model(checkpoint_path, model, mode='weights'):

    if mode == 'weights':
        axial_state_dict = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path)
        axial_state_dict = checkpoint['model_state_dict']

    model.load_state_dict(axial_state_dict)
    
    return model
