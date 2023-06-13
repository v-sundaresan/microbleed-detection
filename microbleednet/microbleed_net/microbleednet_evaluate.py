from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from microbleednet.microbleed_net import microbleednet_data_preparation
from microbleednet.utils import microbleednet_dataset_utils, microbleednet_utils

#=========================================================================================
# Microbleednet evaluate function
# Vaanathi Sundaresan
# 10-01-2023
#=========================================================================================


def dice_coeff(inp, tar):
    """
    Calculating Dice similarity coefficient
    :param inp: Input tensor
    :param tar: Target tensor
    :return: Dice value (scalar)
    """
    smooth = 1.
    pred_vect = inp.contiguous().view(-1)
    target_vect = tar.contiguous().view(-1)
    intersection = (pred_vect * target_vect).sum()
    dice = (2. * intersection + smooth) / (torch.sum(pred_vect) + torch.sum(target_vect) + smooth)
    return dice


def evaluate_cdet(testdata, model, batch_size, device, verbose=False):
    """
    :param testdata: Dataloader object
    :param model: model
    :param batch_size: int
    :param device: cpu or gpu (.cuda())
    :param verbose: bool, display debug messages
    """
    model.eval()
    nsteps = max(testdata[0].shape[0] // batch_size, 1)
    prob_array = np.array([])
    gen_next_test_batch = microbleednet_utils.batch_generator(testdata, batch_size, shuffle=False)
    with torch.no_grad():
        for i in range(nsteps):
            Xv = next(gen_next_test_batch)
            Xv = Xv.transpose(0, 4, 1, 2, 3)
            if verbose:
                print('Validation dimensions.......................................')
                print(Xv.shape)
            Xv = torch.from_numpy(Xv)
            Xv = Xv.to(device=device, dtype=torch.float32)

            val_pred = model.forward(Xv)
            if verbose:
                print('Validation mask dimensions........')
                print(val_pred.size())
            softmax = nn.Softmax()
            probs = softmax(val_pred)
            probs_nparray = probs.detach().cpu().numpy()

            prob_array = np.concatenate((prob_array, probs_nparray), axis=0) if prob_array.size else probs_nparray

        prob_array = prob_array.transpose(0, 2, 3, 1)
    return prob_array


def evaluate_cdisc_teacher(testdata, model, model_class, batch_size, device, verbose=False):
    """
    :param testdata: ndarray
    :param model: model
    :param model_class: classification model
    :param batch_size: int
    :param device: cpu or gpu (.cuda())
    :param verbose: bool, display debug messages
    """
    model.eval()
    nsteps = max(testdata[0].shape[0] // batch_size, 1)
    prob_array = np.array([])
    pred_class_array = np.array([])
    gen_next_test_batch = microbleednet_utils.batch_generator(testdata, batch_size, shuffle=False)
    with torch.no_grad():
        for i in range(nsteps):
            Xv = next(gen_next_test_batch)
            Xv = Xv.transpose(0, 4, 1, 2, 3)
            if verbose:
                print('Validation dimensions.......................................')
                print(Xv.shape)

            Xv = torch.from_numpy(Xv)
            Xv = Xv.to(device=device, dtype=torch.float32)

            val_int, val_pred = model.forward(Xv)
            if verbose:
                print('Validation mask dimensions........')
                print(val_pred.size())

            val_class_pred = model_class(val_int)
            softmax = nn.Softmax()
            probs = softmax(val_pred)
            pred_class = np.argmax(val_class_pred.cpu().detach().numpy(), axis=1)
            probs_nparray = probs.detach().cpu().numpy()
            prob_array = np.concatenate((prob_array, probs_nparray), axis=0) if prob_array.size else probs_nparray
            pred_class_array = np.concatenate((pred_class_array, pred_class), axis=0) \
                if pred_class_array.size else pred_class

        prob_array = prob_array.transpose(0, 2, 3, 1)
        return prob_array, pred_class_array


def evaluate_cdisc_student(testdata, smodel, batch_size, device, verbose=False):
    """
    :param testdata: ndarray
    :param smodel: model
    :param batch_size: int
    :param device: cpu or gpu (.cuda())
    :param verbose: bool, display debug messages
    """
    smodel.eval()
    nsteps = max(testdata[0].shape[0] // batch_size, 1)
    pred_class_array = np.array([])
    gen_next_test_batch = microbleednet_utils.batch_generator(testdata, batch_size, shuffle=False)
    with torch.no_grad():
        for i in range(nsteps):
            Xv = next(gen_next_test_batch)
            Xv = Xv.transpose(0, 4, 1, 2, 3)
            if verbose:
                print('Validation dimensions.......................................')
                print(Xv.shape)
            Xv = torch.from_numpy(Xv)
            Xv = Xv.to(device=device, dtype=torch.float32)

            val_class_pred = smodel.forward(Xv)
            if verbose:
                print('Validation mask dimensions........')
                print(val_class_pred.size())
            pred_class = np.argmax(val_class_pred.cpu().detach().numpy(), axis=1)
            pred_class_array = np.concatenate((pred_class_array, pred_class), axis=0) \
                if pred_class_array.size else pred_class

    return pred_class_array
