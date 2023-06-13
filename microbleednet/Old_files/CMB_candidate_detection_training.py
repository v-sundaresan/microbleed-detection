#!/usr/bin/env python
#   Copyright (C) 2016 University of Oxford
#   SHBASECOPYRIGHT

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import utils
import loss_functions
import model
import cmb_data_preparation1
import warnings
import glob

warnings.filterwarnings("ignore")


def dice_coeff(inp, tar):
    smooth = 1.
    pred_vect = inp.contiguous().view(-1)
    target_vect = tar.contiguous().view(-1)
    intersection = (pred_vect * target_vect).sum()
    dice = (2. * intersection + smooth) / (torch.sum(pred_vect) + torch.sum(target_vect) + smooth)
    return dice


def train_mednet(train_names, val_names, model, batch_size, num_epochs, device, patch_size=32, fold=0,
                 save_checkpoint=True):
    dir_checkpoint = '/path/to/model/checkpoints/'
    batch_factor = 2 # determines how many images are loaded for training at an iteration
    num_iters = max(len(train_names) // batch_factor, 1)
    losses = []
    losses_val = []
    lrt = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lrt, eps=1e-04)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10], gamma=0.2, last_epoch=-1)
    criterion = valdo_loss_functions.CombinedLoss()
    gstep = 0
    start_epoch = 0
    try:
        if dir_checkpoint is not None:
            ckpt_path = os.path.join(dir_checkpoint, 'tmp_ukbb_' + str(patch_size) + '_' + str(fold+1) + '.pth')
        else:
            ckpt_path = os.path.join(os.getcwd(), 'tmp_ukbb_' + str(patch_size) + '_' + str(fold+1) + '.pth')
        checkpoint_resumetraining = torch.load(ckpt_path)
        model.load_state_dict(checkpoint_resumetraining['model_state_dict'])
        optimizer.load_state_dict(checkpoint_resumetraining['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint_resumetraining['scheduler_state_dict'])
        start_epoch = checkpoint_resumetraining['epoch'] + 1
        loss = checkpoint_resumetraining['loss_train']
        val_score = checkpoint_resumetraining['dice_val']
    except:
        print('Not found any model to load and resume training!', flush=True)
    print('Training started!!.......................................', flush=True)
    for epoch in range(start_epoch, num_epochs):
        print('Epoch number: ' + str(epoch), flush=True)
        model.train()
        running_loss = 0.0
        print('Epoch: ' + str(epoch + 1) + 'starting!..............................', flush=True)
        for i in range(num_iters):
            trainnames = train_names[i * batch_factor:(i + 1) * batch_factor]
            print('Training files names listing...................................', flush=True)
            print(trainnames, flush=True)
            train_data = valdo_cmb_data_preparation1.load_and_prepare_cmb_data_frst_ukbb(train_names, train='train',
                                                                                   ps=patch_size)
            val_data = valdo_cmb_data_preparation1.load_and_prepare_cmb_data_frst_ukbb(val_names, train='test',
                                                                                 ps=patch_size)
            if train_data[0].shape[1] == 64:
                batch_size = 8
            else:
                batch_size = batch_size
            valdata = [val_data[0], val_data[1], val_data[2]]
            traindata = [train_data[0], train_data[1], train_data[2]]
            numsteps = min(traindata[0].shape[0] // batch_size, 400)
            print(numsteps, flush=True)
            # print('Training data description....',flush=True)
            # print(len(traindata[0]),flush=True)
            # print(traindata[0].shape,flush=True)
            # print(traindata[1].shape,flush=True)
            # print(traindata[2].shape,flush=True)
            gen_next_train_batch = valdo_utils.batch_generator(traindata, batch_size, shuffle=True)
            # gen_next_val_batch = batch_generator(valdata, batch_size, shuffle=False)
            for j in range(numsteps):
                print(j, flush=True)
                model.train()
                X, y, pw = next(gen_next_train_batch)
                X = X.transpose(0, 4, 1, 2, 3)
                pix_weights = pw
                # print('Training dimensions.......................................')
                # print(X.shape)
                # print(y.shape)
                optimizer.zero_grad()
                X = torch.from_numpy(X)
                y = torch.from_numpy(y)
                pix_weights = torch.from_numpy(pix_weights)
                X = X.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.double)
                pix_weights = pix_weights.to(device=device, dtype=torch.float32)
                masks_pred = model(X)
                loss = criterion(masks_pred, y, weight=pix_weights)
                running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                del X
                del y
                del pix_weights
                gstep += 1
                if j % 100 == 0:
                    val_score, _ = eval_mednet(valdata, model, batch_size, device)
                    scheduler.step(val_score)
        if dir_checkpoint is not None:
            ckpt_path = os.path.join(dir_checkpoint, 'tmp_ukbb_' + str(patch_size) + '_' + str(fold+1) + '.pth')
        else:
            ckpt_path = os.path.join(os.getcwd(), 'tmp_ukbb_' + str(patch_size) + '_' + str(fold+1) + '.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss_train': loss,
            'dice_val': val_score
        }, ckpt_path)
        if epoch % 5 == 0:
            if save_checkpoint:
                try:
                    os.mkdir(dir_checkpoint)
                except OSError:
                    pass
                torch.save(model.state_dict(),
                           dir_checkpoint + f'CP_epoch{epoch+1}_frst' + '_' + str(patch_size) + '_' + str(
                               fold + 1) + '.pth')
        av_loss = (running_loss / num_iters)
        losses.append(av_loss)
        torch.cuda.empty_cache()
        np.save(dir_checkpoint + 'losses.npy', losses)
    return model


def eval_mednet(testdata, model, batch_size, device):
    model.eval()
    nsteps = max(testdata[0].shape[0] // batch_size, 1)
    prob_array = np.array([])
    gen_next_test_batch = valdo_utils.batch_generator(testdata, batch_size, shuffle=False)
    dice_values = 0
    for i in range(nsteps):
        Xv, yv, pwv = next(gen_next_test_batch)
        Xv = Xv.transpose(0, 4, 1, 2, 3)
        # pix_weightsv = pwv
        # pix_weightsv = torch.from_numpy(pix_weightsv)
        # pix_weightsv = pix_weightsv.to(device=device, dtype=torch.float32)
        Xv = torch.from_numpy(Xv)
        Xv = Xv.to(device=device, dtype=torch.float32)
        val_pred = model(Xv)
        yv = torch.from_numpy(yv)
        yv = yv.to(device=device, dtype=torch.double)
        softmax = nn.Softmax()
        probs = softmax(val_pred)
        probs_vector = probs.contiguous().view(-1, 2)
        mask_vector = (probs_vector[:, 1] > 0.5).double()
        target_vector = yv.contiguous().view(-1)
        dice_val = dice_coeff(mask_vector, target_vector)
        probs1 = probs.cpu()
        probs_np = probs1.detach().numpy()
        del Xv
        del yv
        # del pix_weightsv

        prob_array = np.concatenate((prob_array, probs_np), axis=0) if prob_array.size else probs_np

        dice_values += dice_val
    prob_array = prob_array.transpose(0, 2, 3, 4, 1)
    dice_values = dice_values / (nsteps + 1)
    return dice_values, prob_array


data_path1 = 'path/to/the/data/directory'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dir_checkpoint = 'path/to/the/checkpoint/directory/'

spf = 15
bs = 8
for fold in range(0, 1):
    if fold == 4:
        data_path = data_path1[fold * spf:]
    else:
        data_path = data_path1[fold * spf:(fold + 1) * spf]

    for i, name in enumerate(data_path[:1]):
        test_names = [name]
        print(test_names, flush=True)
        rem_ids = np.setdiff1d(np.arange(len(data_path)), i)
        rem_names = [data_path[ind] for ind in rem_ids]
        train_names, val_names, val_ids = valdo_cmb_data_preparation1.select_train_val_names(data_path, 2)
        print(train_names, flush=True)
        print(val_names, flush=True)

        model = valdo_model.MedNet(n_channels=2, n_classes=2, batch_size=bs, init_channels=64)
        model.to(device=device)
        # model.load_state_dict(torch.load(dir_checkpoint + 'CP_epoch26_frst_48_' + str(fold+1) + '.pth'))
        # model64 = train_mednet(train_names, val_names, model, 8, 30, device, patch_size=64, fold=fold)
        model48 = train_mednet(train_names, val_names, model, bs, 61, device, patch_size=48, fold=fold)
        # model32 = train_mednet(train_names, val_names, model, 8, 30, device, patch_size=32, fold=fold)
        # model24 = train_mednet(train_names, val_names, model, 8, 30, device, patch_size=24, fold=fold)

    torch.cuda.empty_cache()
