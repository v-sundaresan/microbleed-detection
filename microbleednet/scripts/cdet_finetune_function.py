from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from torch import optim

from microbleednet.scripts import utils
from microbleednet.scripts import datasets
from microbleednet.scripts import loss_functions
from microbleednet.scripts import data_preparation
from microbleednet.scripts import cdet_train_function
from microbleednet.scripts import model_architectures as models


################################################################
# Microbleednet Candidate detection model fine_tuning function #
# Vaanathi Sundaresan                                          #
# 10-01-2023                                                   #
################################################################


def main(subjects, finetune_params, perform_augmentation=True, save_checkpoint=True, save_weights=True, save_case='best', verbose=True, model_directory=None, checkpoint_directory=None):
    """
    The main function for fine-tuning the model
    :param subjects: list of dictionaries containing subject filepaths for fine-tuning
    :param finetune_params: dictionary of fine-tuning parameters
    :param perform_augmentation: bool, whether to do data augmentation
    :param save_checkpoint: bool, whether to save checkpoint
    :param save_weights: bool, whether to save weights alone or the full model
    :param save_case: str, condition for saving the CP
    :param verbose: bool, display debug messages
    :param model_directory: str, filepath containing pretrained model
    :param checkpoint_directory: str, filepath for saving the model
    """
    assert len(subjects) >= 5, "Number of distinct subjects for fine-tuning cannot be less than 5"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = finetune_params['Optimizer']  # adam, sgd
    gamma = finetune_params['LR_red_factor']  # scalar (0,1)
    patch_size = finetune_params['Patch_size']
    milestones = finetune_params['LR_Milestones']  # list of integers [1, N]
    train_proportion = finetune_params['Train_prop']  # scale (0,1)
    layers_to_finetune = finetune_params['Finetuning_layers']  # list of numbers [1,8]
    finetune_learning_rate = finetune_params['Finetuning_learning_rate']  # scalar (0,1)

    model = models.CDetNet(n_channels=2, n_classes=2, init_channels=64)
    # model = nn.DataParallel(model)
    model.to(device=device)

    model_name = finetune_params['Modelname']
    try:
        model_path = os.path.join(model_directory, f'{model_name}_cdet_model.pth')
        model = utils.load_model(model_path, model, mode='full_model')
    except:
        try:
            model_path = os.path.join(model_directory, f'{model_name}_cdet_model_weights.pth')
            model = utils.load_model(model_path, model, mode='weights')
        except ImportError:
            raise ImportError(f'In directory {model_directory}, {model_name}_cdet_model.pth or {model_name}_cdet_student_model_weights.pth do not appear to be valid model or weights files')

    if type(milestones) != list:
        milestones = [milestones]

    if type(layers_to_finetune) != list:
        layers_to_finetune = [layers_to_finetune]

    print(f'Total parameters in candidate detection model: {sum([p.numel() for p in model.parameters()]) / 1e6} M')

    model = utils.freeze_layers_for_finetuning(model, layers_to_finetune, verbose=verbose)
    model.to(device=device)

    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print(f'Total trainable parameters in candidate detection model: {sum([p.numel() for p in trainable_parameters]) / 1e6} M')

    if optimizer == 'adam':
        epsilon = finetune_params['Epsilon']
        optimizer = optim.Adam(trainable_parameters, lr=finetune_learning_rate, eps=epsilon)
    elif optimizer == 'sgd':
        moment = finetune_params['Momentum']
        optimizer = optim.SGD(trainable_parameters, lr=finetune_learning_rate, momentum=moment)

    criterion = loss_functions.CombinedLoss()

    if verbose:
        print(f'Found {len(subjects)} subjects')

    positive_patches_store, negative_patches_store = data_preparation.split_into_nonoverlapping_patches_classwise(subjects, patch_size=patch_size)

    if verbose:
        print(f"Num positive patches: {len(positive_patches_store)}, Num negative patches: {len(negative_patches_store)}")

    train_patches_store, validation_patches_store = utils.split_patches(positive_patches_store, negative_patches_store, train_proportion)
    train_set = datasets.CDetPatchDataset(train_patches_store['positive'], train_patches_store['negative'], perform_augmentations=True)
    validation_set = datasets.CDetPatchDataset(validation_patches_store['positive'], validation_patches_store['negative'])

    if verbose:
        print(f'Num training patches: {len(train_set)}, Num validation patches: {len(validation_set)}')

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma, last_epoch=-1)

    model = cdet_train_function.train(train_set, validation_set, model, criterion, optimizer, scheduler, finetune_params, device, perform_augmentation=perform_augmentation, save_checkpoint=save_checkpoint, save_weights=save_weights, save_case=save_case, verbose=verbose, checkpoint_directory=checkpoint_directory)
