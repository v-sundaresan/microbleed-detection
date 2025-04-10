from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from re import I
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from microbleednet.scripts import utils

from microbleednet.scripts import data_preparation
from microbleednet.scripts import model_architectures as models

########################################
# Microbleednet main training function #
# Vaanathi Sundaresan                  #
# 09-01-2023                           #
########################################

def main(subjects, verbose=True, model_directory=None, model_name='microbleednet'):
    
    """
    The main evaluation function
    :param subjects: list of dictionaries containing training filepaths
    :param training_params: dictionary of training parameters
    :param perform_augmentation: bool, perform data augmentation
    :param save_checkpoint: bool, save checkpoints
    :param save_weights: bool, if False, the whole model will be saved
    :param save_case: str, condition for saving the checkpoint
    :param verbose: bool, display debug messages
    :param checkpoint_directory: str, directory for saving model/weights
    :return: trained model
    """

    return_type = 'list'
    if type(subjects) != list:
        subjects = [subjects]
        return_type = 'item'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.CDiscStudentNet(n_channels=2, n_classes=2, init_channels=64)
    # model = nn.DataParallel(model)
    model = model.to(device)

    # Load candidate detection model
    try:
        model_path = os.path.join(model_directory, f'{model_name}_cdisc_student_model.pth')
        model = utils.load_model(model_path, model, mode='full_model')
    except:
        try:
            model_path = os.path.join(model_directory, f'{model_name}_cdisc_student_model_weights.pth')
            model = utils.load_model(model_path, model, mode='weights')
        except ImportError:
            raise ImportError(f'In directory, {model_directory}, {model_name}_cdisc_model.pth or {model_name}_cdisc_student_model_weights.pth does not appear to be a valid model file.')

    softmax = nn.Softmax(dim=1) 

    patch_size = 24 # Make this editable
    # patch_size = training_params['Patch_size']

    model.eval()

    for subject in tqdm(subjects, desc='evaluating_cdisc', disable=True):

        image, _, frst, _ = data_preparation.load_subject(subject)
        # Here we assume that the subject has been passed through the CDet evaluation function first.
        try:
            cdet_prediction = subject['cdet_inference']
        except:
            raise ValueError(f"Subject {subject['basename']} has not been evaluated with CDet. Please evaluate with CDet first.")

        data_patches, _ = data_preparation.get_patches_centered_on_cmb(image, cdet_prediction, frst, patch_size)
        data_patches, _= data_preparation.augment_data(data_patches, np.zeros_like(data_patches), n_augmentations=4)
        data_patches[data_patches < 0] = 0

        with torch.no_grad():

            patch_predictions = []
            for patch in data_patches:

                patch = np.expand_dims(patch, axis=0)
                patch = torch.from_numpy(patch)
                patch = patch.to(device=device, dtype=torch.float)

                predictions = model.forward(patch)
                probabilities = softmax(predictions)

                predictions = np.argmax(probabilities.cpu().numpy(), axis=1)
                patch_predictions.append(predictions.item())

        inferred_subject = data_preparation.filter_predictions_from_volume(cdet_prediction, patch_predictions)
        subject['cdisc_inference'] = inferred_subject

    if return_type == 'item':
        return subjects[0]
    return subjects