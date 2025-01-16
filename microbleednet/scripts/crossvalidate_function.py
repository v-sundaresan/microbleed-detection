from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import nibabel as nib
from tqdm import tqdm

from microbleednet.scripts import utils
from microbleednet.scripts import datasets
from microbleednet.scripts import cdet_train_function, cdisc_train_function, loss_functions
from microbleednet.scripts import cdet_evaluate_function
from microbleednet.scripts import cdisc_evaluate_function

import microbleednet.scripts.model_architectures as models
import microbleednet.scripts.data_preparation as data_preparation


###########################################
# Microbleednet cross-validation function #
# Vaanathi Sundaresan                     #
# 09-01-2023                              #
###########################################


def main(subjects, crossvalidation_params, model_directory=None, perform_augmentation=True, intermediate=False, save_checkpoint=False, save_weights=True, save_case='best', verbose=True, checkpoint_directory=None, output_directory=None):
    """
    The main function for leave-one-out validation of Truenet
    :param sub_name_dicts: list of dictionaries containing subject filepaths
    :param cv_params: dictionary of LOO paramaters
    :param model_dir: str, filepath for leading the teacher model
    :param aug: bool, whether to do data augmentation
    :param intermediate: bool, whether to save intermediate results
    :param save_cp: bool, whether to save checkpoint
    :param save_wei: bool, whether to save weights alone or the full model
    :param save_case: str, condition for saving the CP
    :param verbose: bool, display debug messages
    :param dir_cp: str, filepath for saving the model
    :param output_dir: str, filepath for saving the output predictions
    """

    assert len(subjects) >= 5, "Number of distinct subjects for Leave-one-out validation cannot be less than 5"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cdet_model = models.CDetNet(n_channels=2, n_classes=2, init_channels=64)
    cdisc_student_model = models.CDiscStudentNet(n_channels=2, n_classes=2, init_channels=64)

    # cdet_model = nn.DataParallel(cdet_model)
    # cdisc_student_model = nn.DataParallel(cdisc_student_model)

    cdet_model.to(device=device)
    cdisc_student_model.to(device=device)

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

    folds = crossvalidation_params['fold']  # scalar [1, N]
    res_fold = crossvalidation_params['res_fold'] - 1 # scalar [1, N]
    optimizer = crossvalidation_params['Optimizer']  # adam, sgd
    gamma = crossvalidation_params['LR_red_factor']  # scalar (0,1)
    milestones = crossvalidation_params['LR_Milestones']  # list of integers [1, N]
    learning_rate = crossvalidation_params['Learning_rate']  # scalar (0,1)
    train_proportion = crossvalidation_params['Train_prop']  # scalar (0,1)

    if type(milestones) != list:
        milestones = [milestones]

    subjects_per_fold = max(int(np.round(len(subjects) / folds)), 1)

    if optimizer == 'adam':
        epsilon = crossvalidation_params['Epsilon']
        optimizer_cdet = optim.Adam(filter(lambda p: p.requires_grad, cdet_model.parameters()), lr=learning_rate, eps=epsilon)
        optimizer_cdisc = optim.Adam(filter(lambda p: p.requires_grad, cdisc_student_model.parameters()), lr=learning_rate, eps=epsilon)
    elif optimizer == 'sgd':
        moment = crossvalidation_params['Momentum']
        optimizer_cdet = optim.SGD(filter(lambda p: p.requires_grad, cdet_model.parameters()), lr=learning_rate, momentum=moment)
        optimizer_cdisc = optim.SGD(filter(lambda p: p.requires_grad, cdisc_student_model.parameters()), lr=learning_rate, momentum=moment)

    criterion = loss_functions.CombinedLoss()
    distillation_criterion = loss_functions.DistillationLoss()

    if verbose:
        print(f'Found {len(subjects)} subjects')

    for fold in tqdm(range(res_fold, folds)):

        if verbose:
            print(f'Training models for fold #{fold + 1}:')

        if fold == (folds - 1):
            test_subject_ids = np.arange(fold * subjects_per_fold, len(subjects))
            test_subjects = [subjects[i] for i in test_subject_ids]
        else:
            test_subject_ids = np.arange(fold * subjects_per_fold, (fold+1) * subjects_per_fold)
            test_subjects = [subjects[i] for i in test_subject_ids]

        remaining_subject_ids = np.setdiff1d(np.arange(len(subjects)), test_subject_ids)
        remaining_subjects = [subjects[id] for id in remaining_subject_ids]

        if save_checkpoint:
            fold_checkpoint_directory = os.path.join(checkpoint_directory, f'fold{fold}_models')
            os.makedirs(fold_checkpoint_directory, exist_ok=True)

        # Prepping subjects for cdet_train_function
        positive_patches_store, negative_patches_store = data_preparation.split_into_nonoverlapping_patches_classwise(remaining_subjects, patch_size=48)

        train_patches_store, validation_patches_store = utils.split_patches(positive_patches_store, negative_patches_store, train_proportion)
        train_set = datasets.CDetPatchDataset(train_patches_store['positive'], train_patches_store['negative'], ratio='1:1', perform_augmentations=True)
        validation_set = datasets.CDetPatchDataset(validation_patches_store['positive'], validation_patches_store['negative'], ratio='1:1')

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer_cdet, milestones, gamma=gamma, last_epoch=-1)

        cdet_model = cdet_train_function.train(train_set, validation_set, cdet_model, criterion, optimizer_cdet, scheduler, crossvalidation_params, device, perform_augmentation, save_checkpoint, save_weights, save_case, verbose, checkpoint_directory)

        # Prepping subjects for cdisc_train_function
        subjects = cdet_evaluate_function.main(subjects, verbose=verbose, model_directory=checkpoint_directory)
        tp_patches_store, fp_patches_store = data_preparation.split_into_patches_centered_on_cmb_classwise(subjects, patch_size=24)

        train_patches_store, validation_patches_store = utils.split_patches(tp_patches_store, fp_patches_store, train_proportion)
        train_set = datasets.CDiscPatchDataset(train_patches_store['negative'], train_patches_store['positive'], ratio='1:1', perform_augmentations=True)
        validation_set = datasets.CDiscPatchDataset(validation_patches_store['positive'], validation_patches_store['negative'], ratio='1:1')

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer_cdisc, milestones, gamma=gamma, last_epoch=-1)

        cdisc_student_model = cdisc_train_function.train(train_set, validation_set, teacher_model, teacher_classification_head, cdisc_student_model, criterion, distillation_criterion, optimizer_cdisc, scheduler, crossvalidation_params, device, perform_augmentation, save_checkpoint, save_weights, save_case, verbose, checkpoint_directory)

        if verbose:
            print(f'Predicting outputs for subjects in fold {fold + 1}')

        for subject in tqdm(test_subjects, leave=False):

            image_header = nib.load(subject['input_path']).header
            image, label, frst, _ = data_preparation.load_subject(subject)

            subject = cdet_evaluate_function.main(subject, verbose=False, model_directory=model_directory)

            if intermediate:
                os.makedirs(os.path.join(output_directory, 'cdet_predictions'), exist_ok=True)
                save_path = os.path.join(output_directory, 'cdet_predictions', f"predicted_cdet_microbleednet_{subject['basename']}.nii.gz")
                newhdr = image_header.copy()
                newobj = nib.nifti1.Nifti1Image(subject['cdet_inference'], None, header=newhdr)
                nib.save(newobj, save_path)

            subject = cdisc_evaluate_function.main(subject, verbose=verbose, model_directory=model_directory)

            if intermediate:
                os.makedirs(os.path.join(output_directory, 'cdisc_predictions'), exist_ok=True)
                save_path = os.path.join(output_directory, 'cdisc_predictions', f"predicted_cdisc_microbleednet_{subject['basename']}.nii.gz")
                newhdr = image_header.copy()
                newobj = nib.nifti1.Nifti1Image(subject['cdisc_inference'], None, header=newhdr)
                nib.save(newobj, save_path)

            brain_mask = (image > 0).astype(int)
            subject['final_inference'] = data_preparation.shape_based_filtering(subject['cdisc_inference'], brain_mask)

            if intermediate:
                os.makedirs(os.path.join(output_directory, 'final_predictions'), exist_ok=True)
                save_path = os.path.join(output_directory, 'final_predictions', f"predicted_final_microbleednet_{subject['basename']}.nii.gz")
            else:
                save_path = os.path.join(output_directory, f"predicted_final_microbleednet_{subject['basename']}.nii.gz")
            newhdr = image_header.copy()
            newobj = nib.nifti1.Nifti1Image(subject['final_inference'], None, header=newhdr)
            nib.save(newobj, save_path)

        if verbose:
            print(f'Fold {fold + 1}: complete!')

    if verbose:
        print('Cross-validation done!')
