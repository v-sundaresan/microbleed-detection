from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import numpy as np
from tqdm import tqdm
import nibabel as nib

from microbleednet.scripts import utils
from microbleednet.scripts import cdet_evaluate_function
from microbleednet.scripts import cdisc_evaluate_function

import microbleednet.scripts.model_architectures as models
import microbleednet.scripts.data_preparation as data_preparation

####################################
# Microbleednet main test function #
# Vaanathi Sundaresan              #
# 09-01-2023                       #
####################################

def main(subjects, evaluation_parameters, intermediate=False, model_directory=None, load_case='last', output_directory=None, verbose=False):
    """
    The main function for testing Truenet

    :param subjects: list of dictionaries containing subject filepaths
    :param evaluation_parameters: dictionary of evaluation parameters
    :param intermediate: bool, whether to save intermediate results
    :param checkpoint_directory: str, filepath containing the test model
    :param load_case: str, condition for loading the checkpoint
    :param output_directory: str, filepath for saving the output predictions
    :param verbose: bool, display debug messages
    """

    assert len(subjects) > 0, "There must be at least 1 subject for testing."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cdet_model = models.CDetNet(n_channels=2, n_classes=2, init_channels=64)
    cdisc_student_model = models.CDiscStudentNet(n_channels=2, n_classes=2, init_channels=64)

    # cdet_model = nn.DataParallel(cdet_model)
    # cdisc_student_model = nn.DataParallel(cdisc_student_model)

    cdet_model.to(device=device)
    cdisc_student_model.to(device=device)

    model_name = evaluation_parameters['Modelname']
        
    # Load candidate discriminator student model
    try:
        student_model_path = os.path.join(model_directory, f'{model_name}_cdisc_student_model.pth')
        cdisc_student_model = utils.load_model(student_model_path, cdisc_student_model)
    except:
        try:
            student_model_path = os.path.join(model_directory, f'{model_name}_cdisc_student_model_weights.pth')
            cdisc_student_model = utils.load_model(student_model_path, cdisc_student_model, mode='weights')
        except ImportError:
            raise ImportError(f'In directory, {model_directory}, {model_name}_cdisc_student_model.pth or {model_name}_cdisc_student_model_weights.pth does not appear to be a valid model file.')

    if verbose:
        # print('Loaded CDisc weights')
        print(f'Found {len(subjects)} subjects')
        
    for subject in tqdm(subjects, leave=False, desc='evaluating_subjects', disable=True):

        image_header = nib.load(subject['input_path']).header
        image_affine = nib.load(subject['input_path']).affine
        raw_image_shape = nib.load(subject['input_path']).get_fdata().shape
        image, label, frst, crop_coords = data_preparation.load_subject(subject)

        if intermediate:
            os.makedirs(os.path.join(output_directory, 'input_images'), exist_ok=True)
            save_path = os.path.join(output_directory, 'input_images', f"input_microbleednet_{subject['basename']}.nii.gz")
            newhdr = image_header.copy()
            newaff = image_affine.copy()
            image_to_save = data_preparation.replace_into_volume_shape(raw_image_shape, image, crop_coords)
            newobj = nib.nifti1.Nifti1Image(image_to_save, affine=newaff, header=newhdr)
            nib.save(newobj, save_path)

        subject = cdet_evaluate_function.main(subject, verbose=False, model_directory=model_directory, model_name=model_name)

        if intermediate:
            os.makedirs(os.path.join(output_directory, 'cdet_predictions'), exist_ok=True)
            save_path = os.path.join(output_directory, 'cdet_predictions', f"predicted_cdet_microbleednet_{subject['basename']}.nii.gz")
            newhdr = image_header.copy()
            newaff = image_affine.copy()
            image_to_save = data_preparation.replace_into_volume_shape(raw_image_shape, subject['cdet_inference'], crop_coords)
            newobj = nib.nifti1.Nifti1Image(image_to_save, affine=newaff, header=newhdr)
            nib.save(newobj, save_path)

        subject = cdisc_evaluate_function.main(subject, verbose=verbose, model_directory=model_directory, model_name=model_name)

        if intermediate:
            os.makedirs(os.path.join(output_directory, 'cdisc_predictions'), exist_ok=True)
            save_path = os.path.join(output_directory, 'cdisc_predictions', f"predicted_cdisc_microbleednet_{subject['basename']}.nii.gz")
            newhdr = image_header.copy()
            newaff = image_affine.copy()
            image_to_save = data_preparation.replace_into_volume_shape(raw_image_shape, subject['cdisc_inference'], crop_coords)
            newobj = nib.nifti1.Nifti1Image(image_to_save, affine=newaff, header=newhdr)
            nib.save(newobj, save_path)

        brain_mask = (image > 0).astype(int)
        subject['final_inference'] = data_preparation.shape_based_filtering(subject['cdisc_inference'], brain_mask)

        if intermediate:
            os.makedirs(os.path.join(output_directory, 'final_predictions'), exist_ok=True)
            save_path = os.path.join(output_directory, 'final_predictions', f"predicted_final_microbleednet_{subject['basename']}.nii.gz")
        else:
            save_path = os.path.join(output_directory, f"predicted_final_microbleednet_{subject['basename']}.nii.gz")

        newhdr = image_header.copy()
        newaff = image_affine.copy()
        image_to_save = data_preparation.replace_into_volume_shape(raw_image_shape, subject['final_inference'], crop_coords)
        newobj = nib.nifti1.Nifti1Image(image_to_save, affine=newaff, header=newhdr)
        nib.save(newobj, save_path)

    if verbose:
        print('Testing complete for all subjects!', flush=True)













