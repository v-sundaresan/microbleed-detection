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
import torch.nn.functional as F
import os
import utils
import loss_functions
import student_model24
import model_class
import cmb_data_preparation1
import warnings
import glob
from collections import OrderedDict


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


def load_and_combine_model_weights(student, teacher_state_dict, teacher_class_state_dict):
    # teacher: inpconv, convfirst, down1, down2
    # teacher_class: convfirst, down1, down2, fc1, fc2, fc3
    # student: inpconv, convfirst, down1, down2, classconvfirst, classdown1, classdown2, fc1, fc2, fc3
    student_state_dict = OrderedDict()
    for key, value in teacher_state_dict.items():
        print(key)
        if 'inpconv' in key[:8]:
            name = key
        elif 'convfirst' in key[:10]:
            name = key
        elif 'down1' in key[:6]:
            name = key
        elif 'down2' in key[:6]:
            name = key
        else:
            continue
        student_state_dict[name] = value
    for key, value in teacher_class_state_dict.items():
        print(key)
        if 'convfirst' in key[:10]:
            name = 'class' + key
        elif 'down1' in key[:6]:
            name = 'class' + key
        elif 'down2' in key[:6]:
            name = 'class' + key
        elif 'fc1' in key[:4]:
            name = key
        elif 'fc2' in key[:4]:
            name = key
        elif 'fc3' in key[:4]:
            name = key
        else:
            continue
        student_state_dict[name] = value
    student.load_state_dict(student_state_dict)
    return student


def freeze_layer_for_finetuning(model, layer_to_ft, verbose=False):
    '''
    Unfreezing specific layers of the model for fine-tuning
    :param model: model
    :param layer_to_ft: list of ints, layers to fine-tune starting from the decoder end.
    :param verbose: bool, display debug messages
    :return: model after unfreezing only the required layers
    '''
    model_layer_names = ['inpconv', 'convfirst', 'down1', 'down2', 'classconvfirst',
                         'classdown1', 'classdown2', 'fc1', 'fc2', 'fc3']
    model_layers_tobe_ftd = []
    for layer_id in layer_to_ft:
        model_layers_tobe_ftd.append(model_layer_names[layer_id - 1])

    for name, child in model.named_children():
        if name in model_layers_tobe_ftd:
            if verbose:
                print('Model parameters', flush=True)
                print(name + ' is unfrozen', flush=True)
            for param in child.parameters():
                param.requires_grad = True
        else:
            if verbose:
                print('Model parameters', flush=True)
                print(name + ' is frozen', flush=True)
            for param in child.parameters():
                param.requires_grad = False

    return model


def distillation(y, teacher_scores, labels, T, alpha):
    p = F.log_softmax(y/T, dim=1)
    q = F.softmax(teacher_scores/T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) * (T**2) / y.shape[0]
    l_ce = F.cross_entropy(y, labels)
    return l_kl * alpha + l_ce * (1. - alpha)


def train_med_studentnet(train_names, val_names, teacher_model, batch_size, num_epochs, device,
                         patch_size=32, fold=0, save_checkpoint=True):
    dir_checkpoint = '/path/to/model/checkpoints/'
    teacher_model = valdo_student_model24.MedStudentNet(n_channels=2, n_classes=2, batch_size=bs, init_channels=64)
    teacher_model.to(device=device)
    if fold == 4:
        model_state_dict = torch.load(dir_checkpoint + 'CP_epoch46_frst_24_' + str(fold+1) + '.pth')
        model_class_state_dict = torch.load(dir_checkpoint + 'CP_epoch46_class_24_' + str(fold+1) + '.pth')
    else:
        try:
            model_state_dict = torch.load(dir_checkpoint + 'CP_epoch51_frst_24_' + str(fold+1) + '.pth')
            model_class_state_dict = torch.load(dir_checkpoint + 'CP_epoch51_class_24_' + str(fold+1) + '.pth')
        except:
            model_state_dict = torch.load(dir_checkpoint + 'CP_epoch56_frst_24_' + str(fold+1) + '.pth')
            model_class_state_dict = torch.load(dir_checkpoint + 'CP_epoch56_class_24_' + str(fold+1) + '.pth')

    teacher_model = load_and_combine_model_weights(teacher_model, model_state_dict, model_class_state_dict)
    # print('Total number of axial parameters: ', str(sum([p.numel() for p in student_model.parameters()])),
    #       flush=True)

    # model = freeze_layer_for_finetuning(model, [5, 6, 7, 8, 9, 10])
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([p.numel() for p in model_parameters])
    # print('Total number of trainable axial parameters: ', str(params), flush=True)

    student_model = valdo_student_model24.MedStudentNet(n_channels=2, n_classes=2, batch_size=bs, init_channels=64)
    student_model.to(device=device)
    batch_factor = 2  # determines how many images are loaded for training at an iteration
    num_iters = max(len(train_names) // batch_factor, 1)
    losses = []
    losses_val = []
    lrt = 0.001
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lrt, eps=1e-04)
    optimizer = optim.Adam(student_model.parameters(), lr=lrt, eps=1e-04)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10], gamma=0.2, last_epoch=-1)
    criterion = valdo_loss_functions.CombinedLoss() #nn.CrossEntropyLoss()
    gstep = 0
    start_epoch = 0
    try:
        if dir_checkpoint is not None:
            ckpt_path = os.path.join(dir_checkpoint, 'tmp_model24_student_' + str(patch_size) + '.pth')
        else:
            ckpt_path = os.path.join(os.getcwd(), 'tmp_model24_student_' + str(patch_size) + '.pth')
        checkpoint_resumetraining = torch.load(ckpt_path)
        teacher_model.load_state_dict(checkpoint_resumetraining['model_state_dict'])
        optimizer.load_state_dict(checkpoint_resumetraining['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint_resumetraining['scheduler_state_dict'])
        start_epoch = checkpoint_resumetraining['epoch'] + 1
        class_loss = checkpoint_resumetraining['loss_train']
        val_score = checkpoint_resumetraining['acc_val']
    except:
        print('Not found any model to load and resume training!', flush=True)
    print('Training started!!.......................................', flush=True)
    for epoch in range(start_epoch, num_epochs):
        print('Epoch number: ' + str(epoch), flush=True)
        student_model.train()
        teacher_model.eval()
        running_class_loss = 0.0
        running_dist_loss = 0.0
        print('Epoch: ' + str(epoch + 1) + 'starting!..............................', flush=True)
        for i in range(num_iters):
            trainnames = train_names[i * batch_factor:(i + 1) * batch_factor]
            print('Training files names listing...................................', flush=True)
            print(trainnames, flush=True)
            train_data = valdo_cmb_data_preparation1.getting_cmb_data_disc_train(train_names, patch_size=24)

            val_data = valdo_cmb_data_preparation1.getting_cmb_data_disc_train(val_names, patch_size=24)
            if train_data[0].shape[1] == 64:
                batch_size = 8
            else:
                batch_size = batch_size
            valdata = [val_data[0], val_data[1]]
            traindata = [train_data[0], train_data[1]]
            numsteps = min(traindata[0].shape[0] // batch_size, 400)
            print(numsteps, flush=True)
            gen_next_train_batch = valdo_utils.batch_generator(traindata, batch_size, shuffle=True)
            # gen_next_val_batch = batch_generator(valdata, batch_size, shuffle=False)
            for j in range(numsteps):
                print(j, flush=True)
                student_model.train()
                X, y = next(gen_next_train_batch)
                X = X.transpose(0, 4, 1, 2, 3)
                y_class = y[:, 1]
                class_weights = y_class * 100
                class_weights = torch.from_numpy(class_weights)
                class_weights = class_weights.to(device=device, dtype=torch.float32)
                optimizer.zero_grad()
                X = torch.from_numpy(X)
                y = torch.from_numpy(y)

                X = X.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.double)
                teacherscores = teacher_model(X)
                yclass_pred = student_model(X)
                del X
                del y

                y_class = torch.from_numpy(y_class)
                y_class = y_class.to(device=device, dtype=torch.double)
                class_loss = criterion(yclass_pred, y_class, weight=class_weights)
                dist_loss = distillation(yclass_pred, teacherscores, y_class, 4, 0.1)
                total_loss = class_loss + dist_loss
                running_class_loss += class_loss.item()
                running_dist_loss += dist_loss.item()
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                gstep += 1
                if j % 100 == 0:
                    val_score = eval_mednet(valdata, student_model, batch_size, device)
                    scheduler.step(val_score)
        if dir_checkpoint is not None:
            ckpt_path = os.path.join(dir_checkpoint, 'tmp_model24_student_' + str(patch_size) + '.pth')
        else:
            ckpt_path = os.path.join(os.getcwd(), 'tmp_model24_student_' + str(patch_size) + '.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': student_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss_train': class_loss,
            'acc_val': val_score
        }, ckpt_path)
        if epoch % 5 == 0:
            if save_checkpoint:
                try:
                    os.mkdir(dir_checkpoint)
                except OSError:
                    pass
                torch.save(student_model.state_dict(),
                           dir_checkpoint + f'CP_epoch{epoch+1}_student_rboffline' + '_' + str(patch_size) + '_' + str(
                               fold + 1) + '.pth')

        avclass_loss = (running_class_loss / num_iters)
        avdist_loss = (running_dist_loss / num_iters)
        losses.append([avclass_loss,avdist_loss])
        torch.cuda.empty_cache()
        np.save(dir_checkpoint + 'losses.npy', losses)
    return student_model


def eval_mednet(testdata, model, batch_size, device, test=0):
    model.eval()
    if test:
        testdata = append_data(testdata)
    nsteps = max(testdata[0].shape[0] // batch_size, 1)
    gen_next_test_batch = valdo_utils.batch_generator(testdata, batch_size, shuffle=False)
    dice_values = 0
    for i in range(nsteps):
        Xv, yv = next(gen_next_test_batch)
        Xv = Xv.transpose(0, 4, 1, 2, 3)
        Xv = torch.from_numpy(Xv)
        Xv = Xv.to(device=device, dtype=torch.float32)
        val_class_pred = model(Xv)
        y_class = yv[:, 1]
        yv = torch.from_numpy(yv)
        yv = yv.to(device=device, dtype=torch.double)
        del Xv
        del yv
        pred_class = np.argmax(val_class_pred.cpu().detach().numpy(), axis=1)
        dice_val = np.sum(pred_class == y_class) / batch_size

        dice_values += dice_val
    dice_values = dice_values / (nsteps + 1)
    return dice_values


data_path1 = '/path/to/the/input/directory/'
#data_path1 = glob.glob('/gpfs2/well/win/users/tjj573/VALDO_CMB_challenge/Task2/sub-*/*_T2S.nii.gz')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(data_path1)
output_directory = '/path/to/the/output/directory/'
dir_checkpoint = '/path/to/model/checkpoints/'

spf = 14
bs = 8
for fold in range(0, 1):
    if fold == 4:
        data_path = data_path1[fold * spf:]
    else:
        data_path = data_path1[fold * spf:(fold + 1) * spf]

    teacher_model = valdo_model_class.MedNet(n_channels=2, n_classes=2, batch_size=8, init_channels=64)
    teacher_model.to(device=device)
    teacher_model.load_state_dict(torch.load(dir_checkpoint + 'CP_epoch56_frst_48_' + str(fold+1) + '.pth'))

    for i, name in enumerate(data_path[:1]):
        test_names = [name]
        print(test_names, flush=True)
        rem_ids = np.setdiff1d(np.arange(len(data_path)), i)
        rem_names = [data_path[ind] for ind in rem_ids]
        train_names, val_names, val_ids = valdo_cmb_data_preparation1.select_train_val_names(data_path, 2)
        print(train_names, flush=True)
        print(val_names, flush=True)


        model48 = train_med_studentnet(train_names, val_names, teacher_model, bs, 61,
                                       device, patch_size=24, fold=fold)

    torch.cuda.empty_cache()
