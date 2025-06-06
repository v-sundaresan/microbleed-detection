from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch import optim
from microbleednet.old_files.original_microbleed_net import (microbleednet_loss_functions,
                              microbleednet_models, microbleednet_train)
from microbleednet.old_files.utils import microbleednet_utils

#=========================================================================================
# Microbleednet main training function
# Vaanathi Sundaresan
# 09-01-2023
#=========================================================================================


def main(sub_name_dicts, tr_params, aug=True, save_cp=True, save_wei=True, save_case='last',
         verbose=True, dir_cp=None):
    """
    The main training function
    :param sub_name_dicts: list of dictionaries containing training filpaths
    :param tr_params: dictionary of training parameters
    :param aug: bool, perform data augmentation
    :param weighted: bool, apply spatial weights in the loss function
    :param save_cp: bool, save checkpoints
    :param save_wei: bool, if False, the whole model will be saved
    :param save_case: str, condition for saving the checkpoint
    :param verbose: bool, display debug messages
    :param dir_cp: str, directory for saving model/weights
    :return: trained model
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert len(sub_name_dicts) >= 5, "Number of distinct subjects for training cannot be less than 5"

    optim_type = tr_params['Optimizer']  # adam, sgd
    milestones = tr_params['LR_Milestones']  # list of integers [1, N]
    gamma = tr_params['LR_red_factor']  # scalar (0,1)
    lrt = tr_params['Learning_rate']  # scalar (0,1)
    train_prop = tr_params['Train_prop']  # scale (0,1)

    model = microbleednet_models.CDetNet(n_channels=2, n_classes=2, init_channels=64)

    model.to(device=device)
    model = nn.DataParallel(model)

    print('Total number of model parameters to train', flush=True)
    print('CDet model: ', str(sum([p.numel() for p in model.parameters()])), flush=True)

    if optim_type == 'adam':
        epsilon = tr_params['Epsilon']
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lrt, eps=epsilon)
    elif optim_type == 'sgd':
        moment = tr_params['Momentum']
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lrt,
                              momentum=moment)
    else:
        raise ValueError("Invalid optimiser choice provided! Valid options: 'adam', 'sgd'")

    criterion = microbleednet_loss_functions.CombinedLoss()

    if verbose:
        print('Found' + str(len(sub_name_dicts)) + 'subjects', flush=True)

    num_val_subs = max(int(len(sub_name_dicts) * (1 - train_prop)), 1)
    train_name_dicts, val_name_dicts, val_ids = microbleednet_utils.select_train_val_names(sub_name_dicts,
                                                                                     num_val_subs)
    if type(milestones) != list:
        milestones = [milestones]

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma, last_epoch=-1)
    model = microbleednet_train.train_cdet(train_name_dicts, val_name_dicts, model, criterion,
                                           optimizer, scheduler, tr_params, device, augment=aug,
                                           save_checkpoint=save_cp, save_weights=save_wei,
                                           save_case=save_case, verbose=verbose, dir_checkpoint=dir_cp)
    return model


