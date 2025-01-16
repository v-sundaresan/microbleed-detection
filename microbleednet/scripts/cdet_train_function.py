from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader

from microbleednet.scripts import utils
from microbleednet.scripts import datasets
from microbleednet.scripts import earlystopping

from microbleednet.scripts import loss_functions
from microbleednet.scripts import data_preparation
from microbleednet.scripts import model_architectures as models

########################################
# Microbleednet main training function #
# Vaanathi Sundaresan                  #
# 09-01-2023                           #
########################################

def main(subjects, training_params, perform_augmentation=True, save_checkpoint=True, save_weights=True, save_case='last', verbose=True, checkpoint_directory=None):
    
    """
    The main training function
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

    assert len(subjects) >= 5, "Number of distinct subjects for training cannot be less than 5"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gamma = training_params['LR_red_factor']  # scalar (0, 1)
    optimizer = training_params['Optimizer']  # adam, sgd
    patch_size = training_params['Patch_size']
    milestones = training_params['LR_Milestones']  # list of integers [1, N]
    learning_rate = training_params['Learning_rate']  # scalar (0, 1)
    train_proportion = training_params['Train_prop']  # scale (0, 1)

    model = models.CDetNet(n_channels=2, n_classes=2, init_channels=64)
    # model = nn.DataParallel(model)
    model = model.to(device)

    if type(milestones) != list:
        milestones = [milestones]

    if verbose:
        print(f'Total number of model parameters to train in CDet model: {sum([p.numel() for p in model.parameters()]) / 1e6} M')

    trainable_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    if optimizer == 'adam':
        optimizer = optim.Adam(trainable_parameters, lr=learning_rate, eps=training_params['Epsilon'])
    elif optimizer == 'sgd':
        optimizer = optim.SGD(trainable_parameters, lr=learning_rate, momentum=training_params['Momentum'])

    criterion = loss_functions.CombinedLoss()

    if verbose:
        print(f'Found {len(subjects)} subjects')

    positive_patches_store, negative_patches_store = data_preparation.split_into_nonoverlapping_patches_classwise(subjects, patch_size=patch_size)

    if verbose:
        print(f"Num positive patches: {len(positive_patches_store)}, Num negative patches: {len(negative_patches_store)}")

    train_patches_store, validation_patches_store = utils.split_patches(positive_patches_store, negative_patches_store, train_proportion)
    train_set = datasets.CDetPatchDataset(train_patches_store['positive'], train_patches_store['negative'], ratio='1:1', perform_augmentations=True)
    validation_set = datasets.CDetPatchDataset(validation_patches_store['positive'], validation_patches_store['negative'], ratio='1:1')

    if verbose:
        print(f'Num training patches: {len(train_set)}, Num validation patches: {len(validation_set)}')

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma, last_epoch=-1)

    model = train(train_set, validation_set, model, criterion, optimizer, scheduler, training_params, device, perform_augmentation=perform_augmentation, save_checkpoint=save_checkpoint, save_weights=save_weights, save_case=save_case, verbose=verbose, checkpoint_directory=checkpoint_directory)

    return model

def train(train_set, validation_set, model, criterion, optimizer, scheduler, training_params, device, perform_augmentation=True, save_checkpoint=True, save_weights=True, save_case='best', verbose=True, checkpoint_directory=None):
    
    """
    Microbleednet train function
    :param train_set: CDetPatchDataset containing training patches
    :param validation_set: CDetPatchDataset containing validation patches
    :param model: model
    :param criterion: loss function
    :param optimizer: optimiser
    :param scheduler: learning rate scheduler
    :param training_params: dictionary of training parameters
    :param device: cpu() or cuda()
    :param perform_augmentation: bool, perform data augmentation
    :param save_checkpoint: bool
    :param save_weights: bool, if False, whole model will be saved
    :param save_case: str, condition for saving CP
    :param verbose: bool, display debug messages
    :param checkpoint_directory: str, filepath for saving the model
    :return: trained model
    """

    batch_size = training_params['Batch_size']
    num_epochs = training_params['Num_epochs']
    batch_factor = training_params['Batch_factor']
    patience = training_params['Patience']
    save_resume = training_params['SaveResume']
    patch_size = training_params['Patch_size']

    early_stopping = earlystopping.EarlyStoppingModelCheckpointing('cdet', patience, verbose=verbose)

    train_losses = []
    validation_dice = []
    validation_losses = []
    best_validation_dice = 0

    start_epoch = 1
    if save_resume:
        try:
            if checkpoint_directory is not None:
                checkpoint_path = os.path.join(checkpoint_directory, 'tmp_model_cdet.pth')
            else:
                checkpoint_path = os.path.join(os.getcwd(), 'tmp_model_cdet.pth')

            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            train_losses = checkpoint['loss_train']
            validation_losses = checkpoint['loss_val']
            validation_dice = checkpoint['dice_val']
            best_validation_dice = checkpoint['best_val_dice']

        except:
            if verbose:
                print('Not found any model to load and resume training!')

    if verbose:
        print('Starting training.')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    for epoch in range(start_epoch, num_epochs + 1):

        model.train()
        train_set.reset_samples()

        running_loss = 0.0
        print(f'\nEpoch: {epoch}')

        n_batches = len(train_loader)
        with tqdm(total=n_batches, desc='training_cdet', disable=True) as pbar:

            for batch in train_loader:

                x, y, pixel_weights = batch['input'], batch['label'], batch['pixel_weights']
                x = x.reshape(-1, *x.shape[2:])
                y = y.reshape(-1, *y.shape[2:])
                pixel_weights = pixel_weights.reshape(-1, *pixel_weights.shape[2:])

                x = x.to(device=device, dtype=torch.float)
                y = y.to(device=device, dtype=torch.float)
                pixel_weights = pixel_weights.to(device=device, dtype=torch.float)

                optimizer.zero_grad()
                predictions = model.forward(x)
                loss = criterion(predictions, y, weight=pixel_weights)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
                pbar.update(1)

        mean_validation_loss, mean_validation_dice = test(validation_set, model, batch_size, device, criterion, verbose=verbose)
        scheduler.step()

        mean_loss = (running_loss / n_batches)
        mean_loss = mean_loss
        print(f'Training set average loss: {mean_loss:.6f}')

        train_losses.append(mean_loss)
        validation_dice.append(mean_validation_dice.cpu().numpy())
        validation_losses.append(mean_validation_loss)

        if save_resume:
            
            checkpoint_path = os.path.join(os.getcwd(), 'tmp_model_cdet.pth')
            if checkpoint_directory is not None:
                checkpoint_path = os.path.join(checkpoint_directory, 'tmp_model_cdet.pth')
                
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss_train': train_losses,
                'loss_val': validation_losses,
                'dice_val': validation_dice,
                'best_val_dice': best_validation_dice
            }, checkpoint_path)

        if save_checkpoint:
            np.savez(os.path.join(checkpoint_directory, 'losses_cdet.npz'), train_loss=train_losses, val_loss=validation_losses)
            np.savez(os.path.join(checkpoint_directory, 'validation_dice_cdet.npz'), dice_val=validation_dice)

        early_stopping(mean_validation_loss, mean_validation_dice, best_validation_dice, model, epoch, optimizer, scheduler, mean_loss, training_params, weights=save_weights, checkpoint=save_checkpoint, save_condition=save_case, model_path=checkpoint_directory)

        if mean_validation_dice > best_validation_dice:
            best_validation_dice = mean_validation_dice

        if early_stopping.early_stop:
            print('Patience Reached - Early Stopping Activated.')
            break
        # sys.exit('Patience Reached - Early Stopping Activated')

        torch.cuda.empty_cache()  # Clear memory cache

    if save_resume:
        checkpoint_path = os.path.join(os.getcwd(), 'tmp_model_cdet.pth')
        if checkpoint_directory is not None:
            checkpoint_path = os.path.join(checkpoint_directory, 'tmp_model_cdet.pth')
            
        os.remove(checkpoint_path)

    return model

def test(test_set, model, batch_size, device, criterion, verbose=False):
    """
    :param test_data: test_data
    :param model: model
    :param batch_size: int
    :param device: cpu or gpu (.cuda())
    :param criterion: loss function
    :param weighted: bool, whether to apply spatial weights in loss function
    :param verbose: bool, display debug messages
    """

    model.eval()

    softmax = nn.Softmax(dim=1)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    running_loss = 0.0
    running_dice_score = 0

    n_batches = len(test_loader)
    test_set.reset_samples()

    with torch.no_grad():
        with tqdm(total=n_batches, desc='evaluating_cdet', disable=True) as pbar:

            for batch in test_loader:

                x, y, pixel_weights = batch['input'], batch['label'], batch['pixel_weights']
                x = x.reshape(-1, *x.shape[2:])
                y = y.reshape(-1, *y.shape[2:])
                pixel_weights = pixel_weights.reshape(-1, *pixel_weights.shape[2:])

                x = x.to(device=device, dtype=torch.float)
                y = y.to(device=device, dtype=torch.float)
                pixel_weights = pixel_weights.to(device=device, dtype=torch.float)

                predictions = model.forward(x)
                loss = criterion(predictions, y, weight=pixel_weights)
                running_loss += loss.item()
            
                probabilities = softmax(predictions)
                probability_vector = probabilities.view(-1, 2)
                binary_prediction_vector = (probability_vector > 0.5).double()
                target_vector = y.view(-1, 2)

                dice_score = loss_functions.calculate_dice_coefficient(binary_prediction_vector, target_vector)
                running_dice_score += dice_score
            
                pbar.set_postfix({'loss': f'{loss.item():.6f}', 'dice': f'{dice_score:.6f}'})
                pbar.update(1)

    mean_validation_loss = running_loss / n_batches
    mean_validation_dice = running_dice_score / n_batches
 
    print(f'Validation set average loss - {mean_validation_loss:.6f}, Average dice - {mean_validation_dice:.6f}')

    return mean_validation_loss, mean_validation_dice