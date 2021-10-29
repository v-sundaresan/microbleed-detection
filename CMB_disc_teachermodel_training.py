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
import model_class
import cmb_data_preparation1
import cmb_data_postprocessing
import warnings
import glob

warnings.filterwarnings("ignore")
torch.backends.cudnn.enabled = False


def append_data(testdata):
    for d in range(len(testdata)):
        da = testdata[d]
        if len(da.shape) == 4:
            extra = np.zeros([8, da.shape[1], da.shape[2], da.shape[3]])
        elif len(da.shape) == 5:
            extra = np.zeros([8, da.shape[1], da.shape[2], da.shape[3], da.shape[4]])
        else:
            extra = np.zeros([8, da.shape[1], da.shape[2]])
        da = np.concatenate([da, extra], axis=0)
        testdata[d] = da
    return testdata


def dice_coeff(inp, tar):
    smooth = 1.
    pred_vect = inp.contiguous().view(-1)
    target_vect = tar.contiguous().view(-1)
    intersection = (pred_vect * target_vect).sum()
    dice = (2. * intersection + smooth) / (torch.sum(pred_vect) + torch.sum(target_vect) + smooth)
    return dice


def train_mednet(train_names, val_names, model, model_class, batch_size, num_epochs, device, patch_size=32, fold=0,
                 save_checkpoint=True):
    dir_checkpoint = '/path/to/the/initial/candidate/detection/model/checkpoints/'
    batch_factor = 2  # determines how many images are loaded for training at an iteration
    num_iters = max(len(train_names) // batch_factor, 1)
    losses = []
    losses_val = []
    lrt = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lrt, eps=1e-04)
    optimizer_class = optim.Adam(model_class.parameters(), lr=lrt, eps=1e-04)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10], gamma=0.2, last_epoch=-1)
    criterion = valdo_loss_functions.CombinedLoss()
    criterion_class = nn.CrossEntropyLoss()
    gstep = 0
    start_epoch = 0
    try:
        if dir_checkpoint is not None:
            ckpt_path = os.path.join(dir_checkpoint, 'tmp_model24_' + str(patch_size) + '_' + str(fold+1) + '.pth')
            ckpt_path_class = os.path.join(dir_checkpoint, 'tmp_model_class242_' +
                                           str(patch_size) + '_' + str(fold+1) + '.pth')
        else:
            ckpt_path = os.path.join(os.getcwd(), 'tmp_model24_' + str(patch_size) + '_' + str(fold+1) + '.pth')
            ckpt_path_class = os.path.join(os.getcwd(), 'tmp_model_class242_' +
                                           str(patch_size) + '_' + str(fold+1) + '.pth')
        checkpoint_resumetraining = torch.load(ckpt_path)
        model.load_state_dict(checkpoint_resumetraining['model_state_dict'])
        optimizer.load_state_dict(checkpoint_resumetraining['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint_resumetraining['scheduler_state_dict'])
        start_epoch = checkpoint_resumetraining['epoch'] + 1
        loss = checkpoint_resumetraining['loss_train']
        val_score = checkpoint_resumetraining['dice_val']

        checkpoint_resumetraining = torch.load(ckpt_path_class)
        model_class.load_state_dict(checkpoint_resumetraining['model_state_dict'])
        optimizer_class.load_state_dict(checkpoint_resumetraining['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint_resumetraining['scheduler_state_dict'])
        loss_class = checkpoint_resumetraining['loss_class_train']
        val_acc = checkpoint_resumetraining['acc_val']
    except:
        print('Not found any model to load and resume training!', flush=True)
    print('Training started!!.......................................', flush=True)
    for epoch in range(start_epoch, num_epochs):
        print('Epoch number: ' + str(epoch), flush=True) 
        model.train()
        running_loss = 0.0
        running_class_loss = 0.0
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
                y_sum = np.sum(np.sum(np.sum(y, axis=3), axis=2), axis=1)
                y_thr = (y_sum > 0).astype(int)
                y_class = np.zeros([batch_size, 2])
                for m in range(batch_size):
                    y_class[m, y_thr[m]] = 1
                del y_thr
                del y_sum
                class_weights = y_class[:, 1]*100
                class_weights = torch.from_numpy(class_weights)
                class_weights = class_weights.to(device=device, dtype=torch.float32)
                y_class = y_class[:, 1]
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
                int_out, masks_pred = model(X)
                loss = criterion(masks_pred, y, weight=pix_weights)

                del X
                del y
                del pix_weights
                y_class = torch.from_numpy(y_class)
                y_class = y_class.to(device=device, dtype=torch.double)
                yclass_pred = model_class(int_out)
                print(yclass_pred.size())
                print(y_class.size())
                class_loss = criterion(yclass_pred, y_class, weight=class_weights)
                total_loss = loss + class_loss
                running_loss += loss.item()
                running_class_loss += class_loss.item()
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                gstep += 1
                if j % 100 == 0:
                    val_score, _ = eval_mednet(valdata, model, model_class, batch_size, device)
                    scheduler.step(val_score)
        if dir_checkpoint is not None:
            ckpt_path = os.path.join(dir_checkpoint, 'tmp_model24_' + str(patch_size) + '_' + str(fold+1) + '.pth')
            ckpt_path_class = os.path.join(dir_checkpoint, 'tmp_model_class24_' +
                                           str(patch_size) + '_' + str(fold+1) + '.pth')
        else:
            ckpt_path = os.path.join(os.getcwd(), 'tmp_model24_' + str(patch_size) + '_' + str(fold+1) + '.pth')
            ckpt_path_class = os.path.join(os.getcwd(), 'tmp_model_class24_' +
                                           str(patch_size) + '_' + str(fold+1) + '.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss_train': loss,
            'dice_val': val_score
        }, ckpt_path)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_class.state_dict(),
            'optimizer_state_dict': optimizer_class.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss_class_train': class_loss,
            'acc_val': val_score
        }, ckpt_path_class)
        if epoch % 5 == 0:
            if save_checkpoint:
                try:
                    os.mkdir(dir_checkpoint)
                except OSError:
                    pass
                torch.save(model.state_dict(), dir_checkpoint + f'CP_epoch{epoch+1}_teacher_seg' + '_' +
                           str(patch_size) + '_' + str(fold+1) + '.pth')
                torch.save(model_class.state_dict(), dir_checkpoint + f'CP_epoch{epoch+1}_teacher_class' + '_' +
                           str(patch_size) + '_' + str(fold+1) + '.pth')
        av_loss = (running_loss / num_iters)
        avclass_loss = (running_class_loss / num_iters)
        av_loss = [av_loss, avclass_loss]
        losses.append(av_loss)
        torch.cuda.empty_cache()
        np.save(dir_checkpoint + 'losses.npy', losses)
    return model


def eval_mednet(testdata, model, model_class, batch_size, device, test=0):
    model.eval()
    if test:
        testdata = append_data(testdata)
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
        val_int, val_pred = model(Xv)
        val_class_pred = model_class(val_int)
        y_sum = np.sum(np.sum(np.sum(yv, axis=3), axis=2), axis=1)
        y_class = (y_sum > 0).astype(int)
        del y_sum
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
        pred_class = np.argmax(val_class_pred.cpu().detach().numpy(), axis=1)
        dice_val = np.sum(pred_class == y_class)/batch_size
        prob_array = np.concatenate((prob_array, probs_np), axis=0) if prob_array.size else probs_np

        dice_values += dice_val
    prob_array = prob_array.transpose(0, 2, 3, 4, 1)
    dice_values = dice_values / (nsteps + 1)
    return dice_values, prob_array


def eval_multiscale_mednet(testdata24, testdata32, testdata64, model, batch_size, device, test=0):
    model.eval()
    if test:
        testdata24 = append_data(testdata24)
        testdata32 = append_data(testdata32)
        testdata64 = append_data(testdata64)
    print('test data dimensions: ', testdata24[0].shape, testdata32[0].shape, testdata64[0].shape)
    nsteps = max(testdata24[0].shape[0] // batch_size,1)
    prob_array24 = np.array([])
    prob_array32 = np.array([])
    prob_array64 = np.array([])
    gen_next_test_batch24 = valdo_utils.batch_generator(testdata24, batch_size, shuffle=False)
    gen_next_test_batch32 = valdo_utils.batch_generator(testdata32, batch_size, shuffle=False)
    gen_next_test_batch64 = valdo_utils.batch_generator(testdata64, batch_size, shuffle=False)
    dice_values = 0
    for i in range(nsteps):
        Xv, yv, _ = next(gen_next_test_batch24)
        Xv = Xv.transpose(0,4,1,2,3)
        Xv = torch.from_numpy(Xv)
        Xv = Xv.to(device=device, dtype=torch.float32)
        val_pred = model(Xv)
        yv = torch.from_numpy(yv)
        yv = yv.to(device=device, dtype=torch.double)
        softmax = nn.Softmax()
        probs = softmax(val_pred)
        probs_vector = probs.contiguous().view(-1,2)
        mask_vector = (probs_vector[:,1] > 0.5).double()
        target_vector = yv.contiguous().view(-1)
        dice_val24 = dice_coeff(mask_vector, target_vector)
        probs_np = probs.detach().cpu().numpy()
        del Xv
        del yv

        prob_array24 = np.concatenate((prob_array24, probs_np), axis=0) if prob_array24.size else probs_np

        Xv, yv, _ = next(gen_next_test_batch32)
        Xv = Xv.transpose(0, 4, 1, 2, 3)
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
        dice_val32 = dice_coeff(mask_vector, target_vector)
        probs_np = probs.detach().cpu().numpy()
        del Xv
        del yv

        prob_array32 = np.concatenate((prob_array32, probs_np), axis=0) if prob_array32.size else probs_np

        Xv, yv, _ = next(gen_next_test_batch64)
        Xv = Xv.transpose(0, 4, 1, 2, 3)
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
        dice_val64 = dice_coeff(mask_vector, target_vector)
        probs_np = probs.detach().cpu().numpy()
        del Xv
        del yv

        prob_array64 = np.concatenate((prob_array64, probs_np), axis=0) if prob_array64.size else probs_np

        dice_values += (dice_val24 + dice_val32 + dice_val64) / 3
    prob_array24 = prob_array24.transpose(0, 2, 3, 4, 1)
    prob_array32 = prob_array32.transpose(0, 2, 3, 4, 1)
    prob_array64 = prob_array64.transpose(0, 2, 3, 4, 1)
    dice_values = dice_values / (nsteps + 1)
    return dice_values, [prob_array24, prob_array32, prob_array64]


data_path1 = 'path/to/the/input/images/directory'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dir_checkpoint = '/path/to/the/initial/candidate/detection/model/checkpoints/'

spf = 20
bs = 8
for fold in range(1, 2):
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

        model = valdo_model_class.MedNet(n_channels=2, n_classes=2, batch_size=bs, init_channels=64)
        model.to(device=device)
        model_class = valdo_model_class.MedClassNet24(n_channels=2, n_classes=2, batch_size=bs, init_channels=256)
        model_class.to(device=device)
        # model.load_state_dict(torch.load(dir_checkpoint + 'CP_epoch26_frst_48_' + str(fold+1) + '.pth'))
        # model64 = train_mednet(train_names, val_names, model, 8, 30, device, patch_size=64, fold=fold)
        model48 = train_mednet(train_names, val_names, model, model_class, bs, 61, device, patch_size=24, fold=fold)
        # model32 = train_mednet(train_names, val_names, model, 8, 30, device, patch_size=32, fold=fold)
        # model24 = train_mednet(train_names, val_names, model, 8, 30, device, patch_size=24, fold=fold)

    torch.cuda.empty_cache()


