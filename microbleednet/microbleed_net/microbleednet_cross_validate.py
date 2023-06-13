from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import nibabel as nib
from microbleednet.microbleed_net import (microbleednet_loss_functions,
                              microbleednet_models, microbleednet_train, microbleednet_evaluate,
                              microbleednet_data_postprocessing, microbleednet_data_preparation,
                                          microbleednet_data_postpreparation)
from microbleednet.utils import (microbleednet_utils)

#=========================================================================================
# Microbleednet cross-validation function
# Vaanathi Sundaresan
# 09-01-2023
#=========================================================================================


def main(sub_name_dicts, cv_params, model_dir=None, aug=True, intermediate=False, save_cp=False,
         save_wei=True, save_case='best', verbose=True, dir_cp=None, output_dir=None):
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

    assert len(sub_name_dicts) >= 5, "Number of distinct subjects for Leave-one-out validation cannot be less than 5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cdet_model = microbleednet_models.CDetNet(n_channels=2, n_classes=2, init_channels=64)
    cdisc_student_model = microbleednet_models.CDiscStudentNet(n_channels=2, n_classes=2, init_channels=64)

    cdet_model.to(device=device)
    cdisc_student_model.to(device=device)
    cdet_model = nn.DataParallel(cdet_model)
    cdisc_student_model = nn.DataParallel(cdisc_student_model)

    tmodel = microbleednet_models.CDiscNet(n_channels=2, n_classes=2, init_channels=64)
    tmodel.to(device=device)
    tmodel_class = microbleednet_models.CDiscClass24(n_channels=2, n_classes=2, init_channels=256)
    tmodel_class.to(device=device)

    try:
        tmodel_path = os.path.join(model_dir, 'Microbleednet_cdisc_teacher_model.pth')
        tmodel = microbleednet_utils.loading_model(tmodel_path, tmodel, mode='full_model')

        tmodel_class_path = os.path.join(model_dir, 'Microbleednet_cdisc_teacher_class_model.pth')
        tmodel_class = microbleednet_utils.loading_model(tmodel_class_path,
                                                         tmodel_class, mode='full_model')
    except:
        raise ValueError("Invalid saving condition provided! Valid options: best, specific, last")

    optim_type = cv_params['Optimizer']  # adam, sgd
    milestones = cv_params['LR_Milestones']  # list of integers [1, N]
    gamma = cv_params['LR_red_factor']  # scalar (0,1)
    lrt = cv_params['Learning_rate']  # scalar (0,1)
    train_prop = cv_params['Train_prop']  # scalar (0,1)
    fold = cv_params['fold']  # scalar [1, N]
    res_fold = cv_params['res_fold']  # scalar [1, N]

    res_fold = res_fold - 1

    test_subs_per_fold = max(int(np.round(len(sub_name_dicts) / fold)), 1)

    if type(milestones) != list:
        milestones = [milestones]

    if optim_type == 'adam':
        epsilon = cv_params['Epsilon']
        optimizer_cdet = optim.Adam(filter(lambda p: p.requires_grad, cdet_model.parameters()), lr=lrt, eps=epsilon)
        optimizer_cdisc = optim.Adam(filter(lambda p: p.requires_grad, cdisc_student_model.parameters()), lr=lrt, eps=epsilon)
    elif optim_type == 'sgd':
        moment = cv_params['Momentum']
        optimizer_cdet = optim.SGD(filter(lambda p: p.requires_grad, cdet_model.parameters()), lr=lrt,
                                   momentum=moment)
        optimizer_cdisc = optim.SGD(filter(lambda p: p.requires_grad, cdisc_student_model.parameters()), lr=lrt,
                                    momentum=moment)
    else:
        raise ValueError("Invalid optimiser choice provided! Valid options: 'adam', 'sgd'")

    criterion = microbleednet_loss_functions.CombinedLoss()
    criterion_distil = microbleednet_loss_functions.DistillationLoss()

    if verbose:
        print('Found' + str(len(sub_name_dicts)) + 'subjects', flush=True)

    for fld in range(res_fold, fold):
        if verbose:
            print('Training models for fold ' + str(fld+1) + '...', flush=True)

        if fld == (fold - 1):
            test_ids = np.arange(fld * test_subs_per_fold, len(sub_name_dicts))
            test_sub_dicts = sub_name_dicts[test_ids]
        else:
            test_ids = np.arange(fld * test_subs_per_fold, (fld+1) * test_subs_per_fold)
            test_sub_dicts = sub_name_dicts[test_ids]

        rem_sub_ids = np.setdiff1d(np.arange(len(sub_name_dicts)), test_ids)
        rem_sub_name_dicts = [sub_name_dicts[idx] for idx in rem_sub_ids]
        num_val_subs = max(int(len(sub_name_dicts) * (1-train_prop)), 1)
        train_name_dicts, val_name_dicts, val_ids = microbleednet_utils.select_train_val_names(rem_sub_name_dicts,
                                                                                               num_val_subs)
        if save_cp:
            dir_cp = os.path.join(dir_cp, 'fold' + str(fld+1) + '_models')
            os.mkdir(dir_cp)

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer_cdet, milestones, gamma=gamma, last_epoch=-1)
        cdet_model = microbleednet_train.train_cdet(train_name_dicts, val_name_dicts, cdet_model, criterion,
                                                    optimizer_cdet, scheduler, cv_params, device, mode='axial',
                                                    augment=aug, save_checkpoint=save_cp, save_weights=save_wei,
                                                    save_case=save_case, verbose=verbose, dir_checkpoint=dir_cp)

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer_cdisc, milestones, gamma=gamma, last_epoch=-1)
        cdisc_student_model = microbleednet_train.train_cdisc_student(train_name_dicts, val_name_dicts, tmodel,
                                                                      tmodel_class, cdisc_student_model, criterion,
                                                                      criterion_distil, optimizer_cdisc, scheduler,
                                                                      cv_params, device, augment=aug,
                                                                      save_checkpoint=save_cp, save_weights=save_wei,
                                                                      save_case=save_case, verbose=verbose,
                                                                      dir_checkpoint=dir_cp)

        if verbose:
            print('Predicting outputs for subjects in fold ' + str(fld+1) + '...', flush=True)

        for sub in range(len(test_sub_dicts)):
            if verbose:
                print('Predicting for subject ' + str(sub + 1) + '...', flush=True)
            test_sub_dict = [sub_name_dicts[sub]]
            inp_path = test_sub_dict[0]['inp_path']
            inp_hdr = nib.load(inp_path).header
            # load the image here using nibabel
            test_data = microbleednet_data_preparation.load_and_prepare_cmb_data_frst_ukbb(test_sub_dict,
                                                                                           train='test',
                                                                                           ps=48)
            probs_cdet = microbleednet_evaluate.evaluate_cdet(test_data, cdet_model, 8, device, verbose=verbose)
            cdet_output = microbleednet_data_postpreparation.putting_patches_into_images_frst_ukbb(test_sub_dict,
                                                                                                   probs_cdet,
                                                                                                   ps=48)
            if intermediate:
                save_path = os.path.join(output_dir, 'Predicted_cand_det_microbleednet_' + test_sub_dict['basename'] +
                                         '.nii.gz')
                if verbose:
                    print('Saving the intermediate candidate detection ...', flush=True)

                newhdr = inp_hdr.copy()
                newobj = nib.nifti1.Nifti1Image(cdet_output, None, header=newhdr)
                nib.save(newobj, save_path)

            binary_cdet_map = (cdet_output > 0.2).astype(float)
            test_data_class = microbleednet_data_preparation.getting_cmb_data_disc_testing(binary_cdet_map,
                                                                                           test_sub_dict,
                                                                                           patch_size=24)
            probs_cdisc = microbleednet_evaluate.evaluate_cdisc_student(test_data_class, cdisc_student_model, 8,
                                                                        device, verbose=verbose)
            cdisc_output = microbleednet_data_postpreparation.putting_patches_into_images_frst_ukbb(test_sub_dict,
                                                                                                    probs_cdisc,
                                                                                                    ps=24)

            final_output = cdisc_output
            newhdr = inp_hdr.copy()
            newobj = nib.nifti1.Nifti1Image(final_output, None, header=newhdr)
            nib.save(newobj, save_path)

            pred_final = microbleednet_data_postprocessing.get_final_3dvolumes(final_output, test_sub_dict)
            if verbose:
                print('Saving the final prediction ...', flush=True)

            newhdr = inp_hdr.copy()
            newobj = nib.nifti1.Nifti1Image(pred_final, None, header=newhdr)
            nib.save(newobj, save_path)

        if verbose:
            print('Fold ' + str(fld+1) + ': complete!', flush=True)

    if verbose:
        print('Cross-validation done!', flush=True)
