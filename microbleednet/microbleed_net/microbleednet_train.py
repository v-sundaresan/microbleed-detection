from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from microbleednet.true_net import microbleednet_data_preparation
from microbleednet.utils import (microbleednet_dataset_utils, microbleednet_utils)

#=========================================================================================
# Microbleednet training and validation functions
# Vaanathi Sundaresan
# 09-01-2023
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


def validate_cdet(testdata, model, batch_size, device, criterion, verbose=False):
    """
    :param val_dataloader: Dataloader object
    :param model: model
    :param batch_size: int
    :param device: cpu or gpu (.cuda())
    :param criterion: loss function
    :param weighted: bool, whether to apply spatial weights in loss function
    :param verbose: bool, display debug messages
    """
    model.eval()
    nsteps = max(testdata[0].shape[0] // batch_size, 1)
    prob_array = np.array([])
    gen_next_test_batch = microbleednet_utils.batch_generator(testdata, batch_size, shuffle=False)
    dice_values = 0
    dice_values = 0
    val_batch_count = 0
    running_val_loss = 0.0
    with torch.no_grad():
        for i in range(nsteps):
            Xv, yv, pwv = next(gen_next_test_batch)
            Xv = Xv.transpose(0, 4, 1, 2, 3)
            if verbose:
                print('Validation dimensions.......................................')
                print(Xv.shape)
                print(yv.shape)

            Xv = torch.from_numpy(Xv)
            Xv = Xv.to(device=device, dtype=torch.float32)
            yv = torch.from_numpy(yv)
            yv = yv.to(device=device, dtype=torch.double)

            val_pred = model.forward(Xv)
            if verbose:
                print('Validation mask dimensions........')
                print(val_pred.size())
            pwv = torch.from_numpy(pwv)
            pwv = pwv.to(device=device, dtype=torch.float32)
            loss = criterion(val_pred, yv, weight=pwv)
            running_val_loss += loss.item()
            softmax = nn.Softmax()
            probs = softmax(val_pred)
            probs_vector = probs.contiguous().view(-1, 2)
            mask_vector = (probs_vector[:, 1] > 0.5).double()
            target_vector = yv.contiguous().view(-1)
            dice_val = dice_coeff(mask_vector, target_vector)
            dice_values += dice_val
            val_batch_count += 1

    val_av_loss = (running_val_loss / val_batch_count)  # .cpu().numpy()
    val_dice = (dice_values / val_batch_count)  # .detach().cpu().numpy()
    print('Validation set: Average loss: ', val_av_loss, flush=True)
    print('Validation set: Average accuracy: ', val_dice, flush=True)
    return val_av_loss, val_dice


def validate_cdisc_teacher(testdata, model, model_class, batch_size, device, criterion, verbose=False):
    """
    :param testdata: ndarray
    :param model: model
    :param batch_size: int
    :param device: cpu or gpu (.cuda())
    :param criterion: loss function
    :param weighted: bool, whether to apply spatial weights in loss function
    :param verbose: bool, display debug messages
    """
    model.eval()
    nsteps = max(testdata[0].shape[0] // batch_size, 1)
    prob_array = np.array([])
    gen_next_test_batch = microbleednet_utils.batch_generator(testdata, batch_size, shuffle=False)
    dice_values = 0
    dice_values = 0
    val_batch_count = 0
    running_val_loss = 0.0
    with torch.no_grad():
        for i in range(nsteps):
            Xv, yv, pwv = next(gen_next_test_batch)
            Xv = Xv.transpose(0, 4, 1, 2, 3)
            if verbose:
                print('Validation dimensions.......................................')
                print(Xv.shape)
                print(yv.shape)

            Xv = torch.from_numpy(Xv)
            Xv = Xv.to(device=device, dtype=torch.float32)
            yv = torch.from_numpy(yv)
            yv = yv.to(device=device, dtype=torch.double)

            val_int, val_pred = model.forward(Xv)
            if verbose:
                print('Validation mask dimensions........')
                print(val_pred.size())

            val_class_pred = model_class(val_int)
            y_sum = np.sum(np.sum(np.sum(yv, axis=3), axis=2), axis=1)
            y_class = (y_sum > 0).astype(int)
            class_weights = y_class[:, 1] * 100
            class_weights = torch.from_numpy(class_weights)
            class_weights = class_weights.to(device=device, dtype=torch.float32)
            pwv = torch.from_numpy(pwv)
            pwv = pwv.to(device=device, dtype=torch.float32)
            loss = criterion(val_pred, yv, weight=pwv)
            class_loss = loss = criterion(val_class_pred, y_class, weight=class_weights)
            total_loss = loss + class_loss
            running_val_loss += total_loss.item()
            softmax = nn.Softmax()
            probs = softmax(val_pred)
            probs_vector = probs.contiguous().view(-1, 2)
            mask_vector = (probs_vector[:, 1] > 0.5).double()
            target_vector = yv.contiguous().view(-1)
            dice_val = dice_coeff(mask_vector, target_vector)
            dice_values += dice_val
            pred_class = np.argmax(val_class_pred.cpu().detach().numpy(), axis=1)
            acc_val = np.sum(pred_class == y_class) / batch_size

            val_batch_count += 1

    val_av_loss = (running_val_loss / val_batch_count)  # .cpu().numpy()
    val_dice = (dice_values / val_batch_count)  # .detach().cpu().numpy()
    val_acc = (acc_val / val_batch_count)  # .detach().cpu().numpy()
    print('Validation set: Average loss: ', val_av_loss, flush=True)
    print('Validation set: Average accuracy: ', val_dice, flush=True)
    return val_av_loss, val_dice, val_acc


def train_cdet(train_name_dicts, val_name_dicts, model, criterion, optimizer, scheduler, train_params,
               device, augment=True, save_checkpoint=True, save_weights=True, save_case='best',
               verbose=True, dir_checkpoint=None):
    """
    Microbleednet train function
    :param train_name_dicts: list of dictionaries containing training filepaths
    :param val_name_dicts: list of dictionaries containing validation filepaths
    :param model: model
    :param criterion: loss function
    :param optimizer: optimiser
    :param scheduler: learning rate scheduler
    :param train_params: dictionary of training parameters
    :param device: cpu() or cuda()
    :param augment: bool, perform data sugmentation
    :param save_checkpoint: bool
    :param save_weights: bool, if False, whole model will be saved
    :param save_case: str, condition for saving CP
    :param verbose: bool, display debug messages
    :param dir_checkpoint: str, filepath for saving the model
    :return: trained model
    """
    batch_size = train_params['Batch_size']
    num_epochs = train_params['Num_epochs']
    batch_factor = train_params['Batch_factor']
    patience = train_params['Patience']
    aug_factor = train_params['Aug_factor']
    save_resume = train_params['SaveResume']
    patch_size = train_params['Patch_size']

    early_stopping = microbleednet_utils.EarlyStoppingModelCheckpointing(patience, verbose=verbose)

    num_iters = max(len(train_name_dicts) // batch_factor, 1)
    losses_train = []
    losses_val = []
    dice_val = []
    best_val_dice = 0
    gstep = 0

    val_data = microbleednet_data_preparation.load_and_prepare_cmb_data_frst_ukbb(val_name_dicts,
                                                                                  train='test',
                                                                                  ps=patch_size)

    start_epoch = 1
    if save_resume:
        try:
            if dir_checkpoint is not None:
                ckpt_path = os.path.join(dir_checkpoint, 'tmp_model_cdet.pth')
            else:
                ckpt_path = os.path.join(os.getcwd(), 'tmp_model_cdet.pth')
            checkpoint_resumetraining = torch.load(ckpt_path)
            model.load_state_dict(checkpoint_resumetraining['model_state_dict'])
            optimizer.load_state_dict(checkpoint_resumetraining['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_resumetraining['scheduler_state_dict'])
            start_epoch = checkpoint_resumetraining['epoch'] + 1
            losses_train = checkpoint_resumetraining['loss_train']
            losses_val = checkpoint_resumetraining['loss_val']
            dice_val = checkpoint_resumetraining['dice_val']
            best_val_dice = checkpoint_resumetraining['best_val_dice']
        except:
            if verbose:
                print('Not found any model to load and resume training!', flush=True)

    print('Training started!!.......................................')
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0
        print('Epoch: ' + str(epoch) + ' starting!..............................')
        for i in range(num_iters):
            trainnames = train_name_dicts[i * batch_factor:(i + 1) * batch_factor]
            print('Training files names listing...................................')
            print(trainnames)
            train_data = microbleednet_data_preparation.load_and_prepare_cmb_data_frst_ukbb(trainnames,
                                                                                            train='train',
                                                                                            ps=patch_size,
                                                                                            augment=augment)
            if train_data[0].shape[1] == 64:
                batch_size = 8
            valdata = [val_data[0], val_data[1], val_data[2]]
            traindata = [train_data[0], train_data[1], train_data[2]]
            numsteps = min(traindata[0].shape[0] // batch_size, 400)
            print('No. of mini-batches: ' + str(numsteps), flush=True)
            gen_next_train_batch = microbleednet_utils.batch_generator(traindata, batch_size,
                                                                       shuffle=True)

            for j in range(numsteps):
                model.train()
                X, y, pw = next(gen_next_train_batch)
                X = X.transpose(0, 4, 1, 2, 3)
                pix_weights = pw
                if verbose:
                    print('Training dimensions.......................................')
                    print(X.shape)
                    print(y.shape)
                optimizer.zero_grad()
                X = torch.from_numpy(X)
                y = torch.from_numpy(y)
                pix_weights = torch.from_numpy(pix_weights)
                X = X.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.double)
                pix_weights = pix_weights.to(device=device, dtype=torch.float32)
                masks_pred = model.forward(X)
                if verbose:
                    print('mask_pred dimensions........')
                    print(masks_pred.size())
                loss = criterion(masks_pred, y, weight=pix_weights)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                gstep += 1

                if verbose:
                    if j % 100 == 0:
                        print('Train Mini-batch: {} out of Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            (i + 1), epoch, (j + 1) * len(X), traindata[0].shape[0],
                            100. * (j + 1) / traindata[0].shape[0], loss.item()),
                            flush=True)

                batch_count += 1

        val_av_loss, val_av_dice = validate_cdet(valdata, model, batch_size, device, criterion,
                                                 verbose=verbose)
        scheduler.step(val_av_dice)

        av_loss = (running_loss / batch_count)  # .detach().cpu().numpy()
        print('Training set: Average loss: ', av_loss, flush=True)
        losses_train.append(av_loss)
        losses_val.append(val_av_loss)
        dice_val.append(val_av_dice)

        if save_resume:
            if dir_checkpoint is not None:
                ckpt_path = os.path.join(dir_checkpoint, 'tmp_model_cdet.pth')
            else:
                ckpt_path = os.path.join(os.getcwd(), 'tmp_model_cdet.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss_train': losses_train,
                'loss_val': losses_val,
                'dice_val': dice_val,
                'best_val_dice': best_val_dice
            }, ckpt_path)

        if save_checkpoint:
            np.savez(os.path.join(dir_checkpoint, 'losses_cdet.npz'), train_loss=losses_train,
                     val_loss=losses_val)
            np.savez(os.path.join(dir_checkpoint, 'validation_dice_cdet.npz'), dice_val=dice_val)

        early_stopping(val_av_loss, val_av_dice, best_val_dice, model, epoch, optimizer, scheduler, av_loss,
                       train_params, weights=save_weights, checkpoint=save_checkpoint, save_condition=save_case,
                       model_path=dir_checkpoint)

        if val_av_dice > best_val_dice:
            best_val_dice = val_av_dice

        if early_stopping.early_stop:
            print('Patience Reached - Early Stopping Activated', flush=True)
            if save_resume:
                if dir_checkpoint is not None:
                    ckpt_path = os.path.join(dir_checkpoint, 'tmp_model_cdet.pth')
                else:
                    ckpt_path = os.path.join(os.getcwd(), 'tmp_model_cdet.pth')
                os.remove(ckpt_path)
            return model
        #             sys.exit('Patience Reached - Early Stopping Activated')

        torch.cuda.empty_cache()  # Clear memory cache

    if save_resume:
        if dir_checkpoint is not None:
            ckpt_path = os.path.join(dir_checkpoint, 'tmp_model_cdet.pth')
        else:
            ckpt_path = os.path.join(os.getcwd(), 'tmp_model_cdet.pth')
        os.remove(ckpt_path)

    return model


def train_cdisc_teacher(train_name_dicts, val_name_dicts, model, model_class, criterion, optimizer, optimizer_class,
                        scheduler, train_params, device, augment=True, save_checkpoint=True, save_weights=True,
                        save_case='best', verbose=True, dir_checkpoint=None):
    """
    Microbleednet train function
    :param train_name_dicts: list of dictionaries containing training filepaths
    :param val_name_dicts: list of dictionaries containing validation filepaths
    :param model: model
    :param criterion: loss function
    :param optimizer: optimiser
    :param scheduler: learning rate scheduler
    :param train_params: dictionary of training parameters
    :param device: cpu() or cuda()
    :param augment: bool, perform data augmentation
    :param save_checkpoint: bool
    :param save_weights: bool, if False, whole model will be saved
    :param save_case: str, condition for saving CP
    :param verbose: bool, display debug messages
    :param dir_checkpoint: str, filepath for saving the model
    :return: trained model
    """
    batch_size = train_params['Batch_size']
    num_epochs = train_params['Num_epochs']
    batch_factor = train_params['Batch_factor']
    patience = train_params['Patience']
    aug_factor = train_params['Aug_factor']
    save_resume = train_params['SaveResume']

    early_stopping = microbleednet_utils.EarlyStoppingModelCheckpointing(patience, verbose=verbose)

    num_iters = max(len(train_name_dicts) // batch_factor, 1)
    losses_train = []
    losses_val = []
    dice_val = []
    acc_val = []
    best_val_dice = 0
    gstep = 0

    val_data = microbleednet_data_preparation.load_and_prepare_cmb_data_frst_ukbb(val_name_dicts,
                                                                                  train='test',
                                                                                  ps=24)

    start_epoch = 1
    if save_resume:
        try:
            if dir_checkpoint is not None:
                ckpt_path = os.path.join(dir_checkpoint, 'tmp_model_cdisc.pth')
                ckpt_path_class = os.path.join(dir_checkpoint, 'tmp_model_cdisc_class.pth')
            else:
                ckpt_path = os.path.join(os.getcwd(), 'tmp_model_cdisc.pth')
                ckpt_path_class = os.path.join(os.getcwd(), 'tmp_model_cdisc_class.pth')
            checkpoint_resumetraining = torch.load(ckpt_path)
            model.load_state_dict(checkpoint_resumetraining['model_state_dict'])
            optimizer.load_state_dict(checkpoint_resumetraining['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_resumetraining['scheduler_state_dict'])
            start_epoch = checkpoint_resumetraining['epoch'] + 1
            losses_train = checkpoint_resumetraining['total_loss_train']
            losses_val = checkpoint_resumetraining['total_loss_val']
            dice_val = checkpoint_resumetraining['dice_val']
            acc_val = checkpoint_resumetraining['acc_val']
            best_val_dice = checkpoint_resumetraining['best_val_dice']

            checkpoint_resumetraining = torch.load(ckpt_path_class)
            model_class.load_state_dict(checkpoint_resumetraining['model_state_dict'])
            optimizer_class.load_state_dict(checkpoint_resumetraining['optimizer_state_dict'])
        except:
            if verbose:
                print('Not found any model to load and resume training!', flush=True)

    print('Training started!!.......................................')
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0
        print('Epoch: ' + str(epoch) + ' starting!..............................')
        for i in range(num_iters):
            trainnames = train_name_dicts[i * batch_factor:(i + 1) * batch_factor]
            print('Training files names listing...................................')
            print(trainnames)
            train_data = microbleednet_data_preparation.load_and_prepare_cmb_data_frst_ukbb(trainnames,
                                                                                            train='train',
                                                                                            ps=24,
                                                                                            augment=augment)
            if train_data[0].shape[1] == 64:
                batch_size = 8
            valdata = [val_data[0], val_data[1], val_data[2]]
            traindata = [train_data[0], train_data[1], train_data[2]]
            numsteps = min(traindata[0].shape[0] // batch_size, 400)
            print('No. of mini-batches: ' + str(numsteps), flush=True)
            gen_next_train_batch = microbleednet_utils.batch_generator(traindata, batch_size,
                                                                       shuffle=True)

            for j in range(numsteps):
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
                class_weights = y_class[:, 1] * 100
                class_weights = torch.from_numpy(class_weights)
                class_weights = class_weights.to(device=device, dtype=torch.float32)
                y_class = y_class[:, 1]
                pix_weights = pw
                if verbose:
                    print('Training dimensions.......................................')
                    print(X.shape)
                    print(y.shape)
                optimizer.zero_grad()
                X = torch.from_numpy(X)
                y = torch.from_numpy(y)
                pix_weights = torch.from_numpy(pix_weights)
                X = X.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.double)
                pix_weights = pix_weights.to(device=device, dtype=torch.float32)
                int_out, masks_pred = model.forward(X)
                if verbose:
                    print('mask_pred dimensions........')
                    print(masks_pred.size())
                loss = criterion(masks_pred, y, weight=pix_weights)
                y_class = torch.from_numpy(y_class)
                y_class = y_class.to(device=device, dtype=torch.double)
                yclass_pred = model_class(int_out)
                if verbose:
                    print('Classification label dimensions.......................................')
                    print(yclass_pred.size())
                    print(y_class.size())
                class_loss = criterion(yclass_pred, y_class, weight=class_weights)
                total_loss = loss + class_loss
                running_loss += total_loss.item()
                total_loss.backward()
                optimizer.step()
                gstep += 1

                if verbose:
                    if j % 100 == 0:
                        print('Train Mini-batch: {} out of Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            (i + 1), epoch, (j + 1) * len(X), traindata[0].shape[0],
                                            100. * (j + 1) / traindata[0].shape[0], loss.item()),
                            flush=True)

                batch_count += 1

        val_av_loss, val_av_dice, val_av_acc = validate_cdisc_teacher(valdata, model, model_class, batch_size, device, criterion,
                                                                      verbose=verbose)
        scheduler.step(val_av_dice)

        av_loss = (running_loss / batch_count)  # .detach().cpu().numpy()
        print('Training set: Average loss: ', av_loss, flush=True)
        losses_train.append(av_loss)
        losses_val.append(val_av_loss)
        dice_val.append(val_av_dice)
        acc_val.append(val_av_acc)

        if save_resume:
            if dir_checkpoint is not None:
                ckpt_path = os.path.join(dir_checkpoint, 'tmp_model_cdet.pth')
            else:
                ckpt_path = os.path.join(os.getcwd(), 'tmp_model_cdet.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'total_loss_train': losses_train,
                'total_loss_val': losses_val,
                'dice_val': dice_val,
                'acc_val': acc_val,
                'best_val_dice': best_val_dice
            }, ckpt_path)

        if save_checkpoint:
            np.savez(os.path.join(dir_checkpoint, 'losses_cdet.npz'), train_loss=losses_train,
                     val_loss=losses_val)
            np.savez(os.path.join(dir_checkpoint, 'validation_dice_cdet.npz'), dice_val=dice_val)

        early_stopping(val_av_loss, val_av_dice, best_val_dice, model, epoch, optimizer, scheduler, av_loss,
                       train_params, weights=save_weights, checkpoint=save_checkpoint, save_condition=save_case,
                       model_path=dir_checkpoint)

        if val_av_dice > best_val_dice:
            best_val_dice = val_av_dice

        if early_stopping.early_stop:
            print('Patience Reached - Early Stopping Activated', flush=True)
            if save_resume:
                if dir_checkpoint is not None:
                    ckpt_path = os.path.join(dir_checkpoint, 'tmp_model_cdisc.pth')
                    ckpt_path_class = os.path.join(dir_checkpoint, 'tmp_model_cdisc_class.pth')
                else:
                    ckpt_path = os.path.join(os.getcwd(), 'tmp_model_cdisc.pth')
                    ckpt_path_class = os.path.join(os.getcwd(), 'tmp_model_cdisc_class.pth')
                os.remove(ckpt_path)
                os.remove(ckpt_path_class)
            return model
        #             sys.exit('Patience Reached - Early Stopping Activated')

        torch.cuda.empty_cache()  # Clear memory cache

    if save_resume:
        if dir_checkpoint is not None:
            ckpt_path = os.path.join(dir_checkpoint, 'tmp_model_cdisc.pth')
            ckpt_path_class = os.path.join(dir_checkpoint, 'tmp_model_cdisc_class.pth')
        else:
            ckpt_path = os.path.join(os.getcwd(), 'tmp_model_cdisc.pth')
            ckpt_path_class = os.path.join(os.getcwd(), 'tmp_model_cdisc_class.pth')
        os.remove(ckpt_path)
        os.remove(ckpt_path_class)

    return model


def train_cdisc_student(train_name_dicts, val_name_dicts, tmodel, smodel, criterion, optimizer, optimizer_class,
                        scheduler, train_params, device, augment=True, save_checkpoint=True, save_weights=True,
                        save_case='best', verbose=True, dir_checkpoint=None):
    """
    Microbleednet train function
    :param train_name_dicts: list of dictionaries containing training filepaths
    :param val_name_dicts: list of dictionaries containing validation filepaths
    :param model: model
    :param criterion: loss function
    :param optimizer: optimiser
    :param scheduler: learning rate scheduler
    :param train_params: dictionary of training parameters
    :param device: cpu() or cuda()
    :param augment: bool, perform data augmentation
    :param save_checkpoint: bool
    :param save_weights: bool, if False, whole model will be saved
    :param save_case: str, condition for saving CP
    :param verbose: bool, display debug messages
    :param dir_checkpoint: str, filepath for saving the model
    :return: trained model
    """
    batch_size = train_params['Batch_size']
    num_epochs = train_params['Num_epochs']
    batch_factor = train_params['Batch_factor']
    patience = train_params['Patience']
    aug_factor = train_params['Aug_factor']
    save_resume = train_params['SaveResume']

    early_stopping = microbleednet_utils.EarlyStoppingModelCheckpointing(patience, verbose=verbose)

    num_iters = max(len(train_name_dicts) // batch_factor, 1)
    losses_train = []
    losses_val = []
    dice_val = []
    acc_val = []
    best_val_dice = 0
    gstep = 0

    val_data = microbleednet_data_preparation.load_and_prepare_cmb_data_frst_ukbb(val_name_dicts,
                                                                                  train='test',
                                                                                  ps=24)

    start_epoch = 1
    if save_resume:
        try:
            if dir_checkpoint is not None:
                ckpt_path = os.path.join(dir_checkpoint, 'tmp_model_cdisc.pth')
                ckpt_path_class = os.path.join(dir_checkpoint, 'tmp_model_cdisc_class.pth')
            else:
                ckpt_path = os.path.join(os.getcwd(), 'tmp_model_cdisc.pth')
                ckpt_path_class = os.path.join(os.getcwd(), 'tmp_model_cdisc_class.pth')
            checkpoint_resumetraining = torch.load(ckpt_path)
            model.load_state_dict(checkpoint_resumetraining['model_state_dict'])
            optimizer.load_state_dict(checkpoint_resumetraining['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_resumetraining['scheduler_state_dict'])
            start_epoch = checkpoint_resumetraining['epoch'] + 1
            losses_train = checkpoint_resumetraining['total_loss_train']
            losses_val = checkpoint_resumetraining['total_loss_val']
            dice_val = checkpoint_resumetraining['dice_val']
            acc_val = checkpoint_resumetraining['acc_val']
            best_val_dice = checkpoint_resumetraining['best_val_dice']

            checkpoint_resumetraining = torch.load(ckpt_path_class)
            model_class.load_state_dict(checkpoint_resumetraining['model_state_dict'])
            optimizer_class.load_state_dict(checkpoint_resumetraining['optimizer_state_dict'])
        except:
            if verbose:
                print('Not found any model to load and resume training!', flush=True)

    print('Training started!!.......................................')
    for epoch in range(start_epoch, num_epochs + 1):
        smodel.train()