from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from torch import optim

from microbleednet.scripts import utils
from microbleednet.scripts import datasets
from microbleednet.scripts import loss_functions
from microbleednet.scripts import model_architectures as models

from microbleednet.scripts import data_preparation
from microbleednet.scripts import cdisc_train_function
from microbleednet.scripts import cdet_evaluate_function

#####################################################################
# Microbleednet candidate discrimination model fine_tuning function #
# Vaanathi Sundaresan                                               #
# 10-01-2023                                                        #
#####################################################################


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
    milestones = finetune_params['LR_Milestones']  # list of integers [1, N]
    train_proportion = finetune_params['Train_prop']  # scale (0,1)
    layers_to_finetune = finetune_params['Finetuning_layers']  # list of numbers [1,8]
    finetune_learning_rate = finetune_params['Finetuning_learning_rate']  # scalar (0,1)
    patch_size = 24

    student_model = models.CDiscStudentNet(n_channels=2, n_classes=2, init_channels=64)
    # student_model = nn.DataParallel(student_model)
    student_model.to(device=device)

    teacher_model = models.CDiscNet(n_channels=2, n_classes=2, init_channels=64)
    teacher_model.to(device=device)

    teacher_classification_head = models.CDiscClass24(n_channels=2, n_classes=2, init_channels=256)
    teacher_classification_head.to(device=device)

    try:
        try:
            teacher_model_path = os.path.join(model_directory, 'microbleednet_cdisc_teacher_model.pth')
            teacher_model = utils.load_model(teacher_model_path, teacher_model, mode='full_model')
        except:
            try:
                teacher_model_path = os.path.join(model_directory, 'microbleednet_cdisc_teacher_model_weights.pth')
                teacher_model = utils.load_model(teacher_model_path, teacher_model, mode='weights')
            except:
                ValueError('Teacher Discriminator model not loaded correctly.')

        if verbose:
            print('Teacher model loaded.')

        try:
            teacher_classification_head_path = os.path.join(model_directory, 'microbleednet_cdisc_teacher_classification_head.pth')
            teacher_classification_head = utils.load_model(teacher_classification_head_path, teacher_classification_head, mode='full_model')
        except:
            try:
                teacher_classification_head_path = os.path.join(model_directory, 'microbleednet_cdisc_teacher_classification_head_weights.pth')
                teacher_classification_head = utils.load_model(teacher_classification_head_path, teacher_classification_head, mode='weights')
            except:
                ValueError('Teacher Classification head not loaded correctly.')

        if verbose:
            print('Teacher classification head loaded.')

    except:
        raise ValueError('Teacher Discriminator models not loaded correctly.')

    model_name = finetune_params['Modelname']
    try:
        model_path_student = os.path.join(model_directory,  f'{model_name}_cdisc_student_model.pth')
        student_model = utils.load_model(model_path_student, student_model, mode='full_model')
    except:
        try:
            model_path_student = os.path.join(model_directory, f'{model_name}_cdisc_student_model_weights.pth')
            student_model = utils.load_model(model_path_student, student_model, mode='weights')
        except:
            raise ImportError(f'In directory {model_directory}, {model_name}_cdisc_student_model.pth or {model_name}_cdisc_student_model_weights.pth do not appear to be valid model or weights files')

    if type(milestones) != list:
        milestones = [milestones]

    if type(layers_to_finetune) != list:
        layers_to_finetune = [layers_to_finetune]

    print('Total number of model parameters', flush=True)
    print(f'Total parameters in candidate discriminatino model: {sum([p.numel() for p in student_model.parameters()])}')

    student_model = utils.freeze_layers_for_finetuning(student_model, layers_to_finetune, verbose=verbose)
    student_model.to(device=device)

    trainable_parameters = list(filter(lambda p: p.requires_grad, student_model.parameters()))
    print(f'Total trainable parameters in candidate detection model: {sum([p.numel() for p in trainable_parameters]) / 1e6} M')

    if optimizer == 'adam':
        epsilon = finetune_params['Epsilon']
        optimizer = optim.Adam(trainable_parameters, lr=finetune_learning_rate, eps=epsilon)
    elif optimizer == 'sgd':
        moment = finetune_params['Momentum']
        optimizer = optim.SGD(trainable_parameters, lr=finetune_learning_rate, momentum=moment)

    criterion = loss_functions.CombinedLoss()
    distillation_criterion = loss_functions.DistillationLoss()

    if verbose:
        print(f'Found {len(subjects)} subjects')

    # This adds the cdet inference to each subject dictionary
    subjects = cdet_evaluate_function.main(subjects, verbose=verbose, model_directory=checkpoint_directory)
    tp_patches_store, fp_patches_store = data_preparation.split_into_patches_centered_on_cmb_classwise(subjects, patch_size=patch_size)

    if verbose:
        print(f'Num tp patches: {len(tp_patches_store)}, Num fp patches: {len(fp_patches_store)}')

    train_patches_store, validation_patches_store = utils.split_patches(tp_patches_store, fp_patches_store, train_proportion)
    train_set = datasets.CDiscPatchDataset(train_patches_store['negative'], train_patches_store['positive'], perform_augmentations=True)
    validation_set = datasets.CDiscPatchDataset(validation_patches_store['negative'], validation_patches_store['positive'])

    if verbose:
        print(f'Num training patches: {len(train_set)}, Num validation patches: {len(validation_set)}')

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma, last_epoch=-1)

    model = cdisc_train_function.train(train_set, validation_set, teacher_model, teacher_classification_head, student_model, criterion, distillation_criterion, optimizer, scheduler, finetune_params, device, perform_augmentation=perform_augmentation, save_checkpoint=save_checkpoint, save_weights=save_weights, save_case=save_case, verbose=verbose, checkpoint_directory=checkpoint_directory)
