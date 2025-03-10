from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import os
import nibabel as nib
from microbleednet.old_files.original_microbleed_net import (microbleednet_models, microbleednet_evaluate,
                                          microbleednet_data_postprocessing, microbleednet_data_preparation,
                                          microbleednet_data_postpreparation, microbleednet_final_filteringfunctions)
from microbleednet.old_files.utils import microbleednet_utils


# =========================================================================================
# Microbleednet main test function
# Vaanathi Sundaresan
# 09-01-2023
# =========================================================================================

def main(sub_name_dicts, eval_params, intermediate=False, model_dir=None,
         load_case='last', output_dir=None, verbose=False):
    """
    The main function for testing Truenet
    :param sub_name_dicts: list of dictionaries containing subject filepaths
    :param eval_params: dictionary of evaluation parameters
    :param intermediate: bool, whether to save intermediate results
    :param model_dir: str, filepath containing the test model
    :param load_case: str, condition for loading the checkpoint
    :param output_dir: str, filepath for saving the output predictions
    :param verbose: bool, display debug messages
    """
    assert len(sub_name_dicts) > 0, "There must be at least 1 subject for testing."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cdet_model = microbleednet_models.CDetNet(n_channels=2, n_classes=2, init_channels=64)
    cdisc_student_model = microbleednet_models.CDiscStudentNet(n_channels=2, n_classes=2, init_channels=64)

    cdet_model.to(device=device)
    cdisc_student_model.to(device=device)
    cdet_model = nn.DataParallel(cdet_model)
    cdisc_student_model = nn.DataParallel(cdisc_student_model)

    model_name = eval_params['Modelname']
    try:
        model_path = os.path.join(model_dir, model_name + '_cdet_model.pth')
        cdet_model = microbleednet_utils.loading_model(model_path, cdet_model)

        model_path_student = os.path.join(model_dir, model_name + '_cdisc_student_model.pth')
        cdisc_student_model = microbleednet_utils.loading_model(model_path_student, cdisc_student_model)
    except:
        try:
            model_path = os.path.join(model_dir, model_name + '_cdet_model.pth')
            cdet_model = microbleednet_utils.loading_model(model_path, cdet_model, mode='full_model')
            model_path_student = os.path.join(model_dir, model_name + '_cdisc_student_model.pth')
            cdisc_student_model = microbleednet_utils.loading_model(model_path_student, cdisc_student_model,
                                                                    mode='full_model')
        except ImportError:
            raise ImportError('In directory ' + model_dir + ', ' + model_name + '_cdet_model.pth or' +
                              model_name + '_cdisc_student_model.pth does not appear to be a valid model file')

    if verbose:
        print('Found' + str(len(sub_name_dicts)) + 'subjects', flush=True)
    for sub in range(len(sub_name_dicts)):
        if verbose:
            print('Predicting output for subject ' + str(sub + 1) + '...', flush=True)

        test_sub_dict = [sub_name_dicts[sub]]
        inp_path = test_sub_dict[0]['inp_path']
        inp_hdr = nib.load(inp_path).header
        # load the image here using nibabel
        test_data, brain = microbleednet_data_preparation.load_cmb_testdata_frst(test_sub_dict)
        probs_cdet = microbleednet_evaluate.evaluate_cdet(test_data, cdet_model, 1, device, verbose=verbose)
        cdet_output = microbleednet_data_postpreparation.putting_outputs_into_images_frst(test_sub_dict, probs_cdet)
        cdet_output /= np.amax(cdet_output)
        if intermediate:
            save_path = os.path.join(output_dir, 'Predicted_cand_det_microbleednet_' + test_sub_dict[0]['basename'] +
                                     '.nii.gz')
            if verbose:
                print('Saving the intermediate candidate detection ...', flush=True)

            newhdr = inp_hdr.copy()
            newobj = nib.nifti1.Nifti1Image(cdet_output, None, header=newhdr)
            nib.save(newobj, save_path)
        binary_cdet_map = (cdet_output > 0.2).astype(float)
        print(binary_cdet_map.shape)
        test_data_class = microbleednet_data_preparation.getting_cmb_data_disc_testing(binary_cdet_map,
                                                                                       test_sub_dict,
                                                                                       patch_size=24)
        print('test_data_class', test_data_class.shape, flush=True)
        probs_cdisc = microbleednet_evaluate.evaluate_cdisc_student(test_data_class, cdisc_student_model, 1,
                                                                    device, verbose=verbose)
        print(probs_cdisc)
        cdisc_output = microbleednet_data_postpreparation.putting_patches_into_images_frst_ukbb_testing(test_sub_dict,
                                                                                                binary_cdet_map, probs_cdisc,
                                                                                                ps=24)
        if intermediate:
            save_path = os.path.join(output_dir, 'Predicted_cand_disc_microbleednet_' + test_sub_dict[0]['basename'] +
                                     '.nii.gz')
            if verbose:
                print('Saving the intermediate candidate discrimination ...', flush=True)

            newhdr = inp_hdr.copy()
            # newobj = nib.nifti1.Nifti1Image(cdisc_output[0], None, header=newhdr)
            newobj = nib.nifti1.Nifti1Image(cdet_output, None, header=newhdr)
            nib.save(newobj, save_path)
        print('final filtering stage')
        final_output = cdet_output
        cmb_map = microbleednet_final_filteringfunctions.candidate_shapebased_filtering(final_output, brain)
        save_path = os.path.join(output_dir, 'Predicted_microbleednet_final_' + test_sub_dict[0]['basename'] +
                                 '.nii.gz')
        if verbose:
            print('Saving the final prediction ...', flush=True)

        newhdr = inp_hdr.copy()
        newobj = nib.nifti1.Nifti1Image(cmb_map, None, header=newhdr)
        nib.save(newobj, save_path)

    if verbose:
        print('Testing complete for all subjects!', flush=True)













