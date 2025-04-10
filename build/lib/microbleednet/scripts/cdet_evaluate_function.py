from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from re import I
import torch
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

    model = models.CDetNet(n_channels=2, n_classes=2, init_channels=64)
    # model = nn.DataParallel(model)
    model = model.to(device)

    # Load candidate detection model
    try:
        model_path = os.path.join(model_directory, f'{model_name}_cdet_model.pth')
        model = utils.load_model(model_path, model)
    except:
        try:
            model_path = os.path.join(model_directory, f'{model_name}_cdet_model_weights.pth')
            model = utils.load_model(model_path, model, mode='weights')
        except ImportError:
            raise ImportError(f'In directory, {model_directory}, {model_name}_cdet_model.pth or {model_name}_cdet_model_weights.pth does not appear to be a valid model file.')

    if verbose:
        print(f'Loaded CDet to get initial predictions')

    softmax = nn.Softmax(dim=1) 

    patch_size = 48 # Make this editable
    # patch_size = training_params['Patch_size']

    model.eval()

    for subject in tqdm(subjects, desc='evaluating_cdet', disable=True, leave=False):

        image, label, frst, _ = data_preparation.load_subject(subject)
        brain_mask = (image > 0).astype(int)

        data_patches, _, _, _, _ = data_preparation.get_nonoverlapping_patches(image, label, brain_mask, frst, patch_size)
        data_patches[data_patches < 0] = 0

        with torch.no_grad():

            inferred_patches = []
            for patch in data_patches:
                
                patch = np.expand_dims(patch, axis=0)
                patch = torch.from_numpy(patch)
                patch = patch.to(device=device, dtype=torch.float)

                predictions = model.forward(patch)
                predictions = softmax(predictions)
                predictions = predictions[0, 1]

                binary_predictions = (predictions > 0.2)
                inferred_patches.append(binary_predictions.cpu().numpy())
            
        inferred_patches = np.stack(inferred_patches, axis=0)
        inferred_subject = data_preparation.put_patches_into_volume(inferred_patches, label, patch_size)
        I
        subject['cdet_inference'] = inferred_subject

    if return_type == 'item':
        return subjects[0]
    return subjects