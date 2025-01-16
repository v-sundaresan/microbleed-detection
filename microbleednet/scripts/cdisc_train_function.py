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
from microbleednet.scripts import loss_functions
from microbleednet.scripts import model_architectures as models

from microbleednet.scripts import earlystopping
from microbleednet.scripts import data_preparation
from microbleednet.scripts import cdet_evaluate_function

########################################
# Microbleednet main training function #
# Vaanathi Sundaresan                  #
# 09-01-2023                           #
########################################


def main(subjects, training_params, model_directory=None, perform_augmentation=True, save_checkpoint=True, save_weights=True, save_case='last', verbose=True, checkpoint_directory=None):
    
    """
    The main training function

    :param subjects: list of dictionaries containing training filpaths
    :param training_params: dictionary of training parameters
    :param model_directory: str, directory for loading the teacher model
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
    milestones = training_params['LR_Milestones']  # list of integers [1, N]
    learning_rate = training_params['Learning_rate']  # scalar (0, 1)
    train_proportion = training_params['Train_prop']  # scale (0, 1)
    patch_size = 24

    student_model = models.CDiscStudentNet(n_channels=2, n_classes=2, init_channels=64)
    # student_model = nn.DataParallel(student_model)
    student_model.to(device=device)

    teacher_model = models.CDiscNet(n_channels=2, n_classes=2, init_channels=64)
    teacher_model.to(device=device)

    teacher_classification_head = models.CDiscClass24(n_channels=2, n_classes=2, init_channels=256)
    teacher_classification_head.to(device=device)

    try:
        try:
            teacher_model_path = os.path.join(model_directory, 'microbleednet_cdisc_teacher_model.pth')
            teacher_model = utils.load_model(teacher_model_path, teacher_model, mode='full_model')
        except:
            try:
                teacher_model_path = os.path.join(model_directory, 'microbleednet_cdisc_teacher_model_weights.pth')
                teacher_model = utils.load_model(teacher_model_path, teacher_model, mode='weights')
            except:
                ValueError('Teacher Discriminator model not loaded correctly.')

        if verbose:
            print('Teacher model loaded.')

        try:
            teacher_classification_head_path = os.path.join(model_directory, 'microbleednet_cdisc_teacher_classification_head.pth')
            teacher_classification_head = utils.load_model(teacher_classification_head_path, teacher_classification_head, mode='full_model')
        except:
            try:
                teacher_classification_head_path = os.path.join(model_directory, 'microbleednet_cdisc_teacher_classification_head_weights.pth')
                teacher_classification_head = utils.load_model(teacher_classification_head_path, teacher_classification_head, mode='weights')
            except:
                ValueError('Teacher Classification head not loaded correctly.')

        if verbose:
            print('Teacher classification head loaded.')

    except:
        raise ValueError('Teacher Discriminator models not loaded correctly.')
    
    if type(milestones) != list:
        milestones = [milestones]

    if verbose:
        print(f'Total number of model parameters to train in CDisc student model: {sum([p.numel() for p in student_model.parameters()]) / 1e6} M')

    if optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, student_model.parameters()), lr=learning_rate, eps=training_params['Epsilon'])
    elif optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, student_model.parameters()), lr=learning_rate, momentum=training_params['Momentum'])

    criterion = loss_functions.CombinedLoss()
    distillation_criterion = loss_functions.DistillationLoss()

    if verbose:
        print(f'Found {len(subjects)} subjects')

    # This adds the cdet inference to each subject dictionary
    subjects = cdet_evaluate_function.main(subjects, verbose=verbose, model_directory=checkpoint_directory)
    tp_patches_store, fp_patches_store = data_preparation.split_into_patches_centered_on_cmb_classwise(subjects, patch_size=patch_size)

    if verbose:
        print(f'Num tp patches: {len(tp_patches_store)}, Num fp patches: {len(fp_patches_store)}')

    train_patches_store, validation_patches_store = utils.split_patches(tp_patches_store, fp_patches_store, train_proportion)
    train_set = datasets.CDiscPatchDataset(train_patches_store['positive'], train_patches_store['negative'], ratio='1:1', perform_augmentations=True)
    validation_set = datasets.CDiscPatchDataset(validation_patches_store['positive'], validation_patches_store['negative'], ratio='1:1')

    if verbose:
        print(f'Num training patches: {len(train_set)}, Num validation patches: {len(validation_set)}')

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma, last_epoch=-1)

    model = train(train_set, validation_set, teacher_model, teacher_classification_head, student_model, criterion, distillation_criterion, optimizer, scheduler, training_params, device, perform_augmentation=perform_augmentation, save_checkpoint=save_checkpoint, save_weights=save_weights, save_case=save_case, verbose=verbose, checkpoint_directory=checkpoint_directory)

    return model

def train(train_set, validation_set, teacher_model, teacher_classification_head, student_model, criterion, distillation_criterion, optimizer, scheduler, training_params, device, perform_augmentation=True, save_checkpoint=True, save_weights=True, save_case='best', verbose=True, checkpoint_directory=None):
    """
    Microbleednet train function

    :param train_set: CDiscPatchDataset containing training patches
    :param validation_subjects: CDiscPatchDataset containing training patches
    :param teacher_model: model
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

    # batch_size = training_params['Batch_size']
    batch_size = 16
    num_epochs = training_params['Num_epochs']
    batch_factor = training_params['Batch_factor']
    patience = training_params['Patience']
    save_resume = training_params['SaveResume']
    patch_size = training_params['Patch_size']

    early_stopping = earlystopping.EarlyStoppingModelCheckpointing('cdisc_student', patience, verbose=verbose)

    train_losses = []
    validation_dice = []
    validation_losses = []
    validation_accuracy = []
    best_validation_accuracy = 0

    start_epoch = 1

    if save_resume:
        try:
            if checkpoint_directory is not None:
                checkpoint_path = os.path.join(checkpoint_directory, 'tmp_model_cdisc_student.pth')
            else:
                checkpoint_path = os.path.join(os.getcwd(), 'tmp_model_cdisc_student.pth')

            checkpoint_resumetraining = torch.load(checkpoint_path)
            student_model.load_state_dict(checkpoint_resumetraining['model_state_dict'])
            optimizer.load_state_dict(checkpoint_resumetraining['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_resumetraining['scheduler_state_dict'])
            start_epoch = checkpoint_resumetraining['epoch'] + 1
            train_losses = checkpoint_resumetraining['total_loss_train']
            validation_losses = checkpoint_resumetraining['total_loss_val']
            validation_dice = checkpoint_resumetraining['dice_val']
            validation_accuracy = checkpoint_resumetraining['acc_val']
            best_validation_dice = checkpoint_resumetraining['best_val_dice']
        except:
            if verbose:
                print('Not found any model to load and resume training!')

    if verbose:
        print('Starting training.')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    softmax = nn.Softmax(dim=1)

    for epoch in range(start_epoch, num_epochs + 1):

        teacher_model.eval()
        student_model.train()
        train_set.reset_samples()

        running_distillation_loss = 0.0
        running_classification_loss = 0.0

        running_positive_sampled = 0
        running_positive_predictions = 0

        running_negative_sampled = 0
        running_negative_predictions = 0
        
        print(f'\nEpoch: {epoch}')

        n_batches = len(train_loader)
        with tqdm(total=n_batches, desc='training_cdisc', disable=True) as pbar:

            for batch in train_loader:

                x, y = batch['input'], batch['label']
                x = x.reshape(-1, *x.shape[2:])
                y = y.reshape(-1, *y.shape[2:])
                classification_weights = y * 100

                x = x.to(device=device, dtype=torch.float)
                y = y.to(device=device, dtype=torch.float)
                classification_weights = classification_weights.to(device=device, dtype=torch.float)

                optimizer.zero_grad()
                teacher_encodings, _ = teacher_model.forward(x)
                teacher_predictions = teacher_classification_head.forward(teacher_encodings)
                student_predictions = student_model.forward(x)
                
                # classification_loss = criterion(student_predictions, y)
                classification_loss = criterion(student_predictions, y, classification_weights)
                distillation_loss = distillation_criterion(student_predictions, teacher_predictions, y, 4, 0.4)
                total_loss = classification_loss + distillation_loss
                # total_loss = distillation_loss

                running_distillation_loss += distillation_loss.item()
                running_classification_loss += classification_loss.item()

                total_loss.backward()
                optimizer.step()

                probabilities = softmax(student_predictions)
                binary_prediction_vector = np.argmax(probabilities.detach().cpu().numpy(), axis=1)
                target_vector = y[:, 1].cpu().numpy()

                positives_predicted = np.sum(np.logical_and(binary_prediction_vector == 1, target_vector == 1))
                positives_sampled = np.sum(target_vector == 1)

                negatives_predicted = np.sum(np.logical_and(binary_prediction_vector == 0, target_vector == 0))
                negatives_sampled = np.sum(target_vector == 0)

                running_positive_sampled += positives_sampled
                running_positive_predictions += positives_predicted

                running_negative_sampled += negatives_sampled
                running_negative_predictions += negatives_predicted

                pbar.set_postfix({'classification_loss': f'{classification_loss.item():.6f}', 'distillation_loss': f'{distillation_loss.item():.6f}', 'positives predicted': f'({positives_predicted}/{positives_sampled})', 'negatives predicted': f'({negatives_predicted}/{negatives_sampled})'})
                pbar.update(1)


        mean_validation_loss, mean_validation_accuracy = test(validation_set, student_model, batch_size, device, criterion, verbose=verbose)
        scheduler.step()

        mean_classification_loss = (running_classification_loss / n_batches)  # .detach().cpu().numpy()
        mean_distillation_loss = (running_distillation_loss / n_batches)
        mean_loss = [mean_classification_loss, mean_distillation_loss]

        print(f'Training set average loss: {(mean_classification_loss + mean_distillation_loss):.6f}, Positives predicted - ({running_positive_predictions}/{running_positive_sampled}), Negatives predicted - ({running_negative_predictions}/{running_negative_sampled})')

        train_losses.append(mean_loss)
        validation_losses.append(mean_validation_loss)
        validation_accuracy.append(mean_validation_accuracy)

        if save_resume:
            if checkpoint_directory is not None:
                checkpoint_path = os.path.join(checkpoint_directory, 'tmp_model_cdisc_student.pth')
            else:
                checkpoint_path = os.path.join(os.getcwd(), 'tmp_model_cdisc_student.pth')

            torch.save({
                'epoch': epoch,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'total_loss_train': train_losses,
                'total_loss_val': validation_losses,
                'dice_val': validation_dice,
                'acc_val': validation_accuracy,
                'best_val_acc': best_validation_accuracy
            }, checkpoint_path)

        if save_checkpoint:
            np.savez(os.path.join(checkpoint_directory, 'losses_cdisc_student.npz'), train_loss=train_losses, val_loss=validation_losses)
            np.savez(os.path.join(checkpoint_directory, 'validation_acc_cdisc_student.npz'), dice_val=validation_accuracy)

        early_stopping(mean_validation_loss, mean_validation_accuracy, best_validation_accuracy, student_model, epoch, optimizer, scheduler, mean_loss, training_params, weights=save_weights, checkpoint=save_checkpoint, save_condition=save_case, model_path=checkpoint_directory)

        if mean_validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = mean_validation_accuracy

        if early_stopping.early_stop:
            print('Patience Reached - Early Stopping Activated.')
            break
        # sys.exit('Patience Reached - Early Stopping Activated')

        torch.cuda.empty_cache()  # Clear memory cache

    if save_resume:
        if checkpoint_directory is not None:
            checkpoint_path = os.path.join(checkpoint_directory, 'tmp_model_cdisc_student.pth')
        else:
            checkpoint_path = os.path.join(os.getcwd(), 'tmp_model_cdisc_student.pth')
        os.remove(checkpoint_path)

    return student_model

def test(test_set, student_model, batch_size, device, criterion, verbose=False):
    """
    :param test_set: test_set
    :param student_model: model
    :param batch_size: int
    :param device: cpu or gpu (.cuda())
    :param criterion: loss function
    :param weighted: bool, whether to apply spatial weights in loss function
    :param verbose: bool, display debug messages
    """

    student_model.eval()

    softmax = nn.Softmax(dim=1)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    running_loss = 0.0
    running_correct_predictions = 0
    total_samples = 0

    running_positive_sampled = 0
    running_positive_predictions = 0

    running_negative_sampled = 0
    running_negative_predictions = 0

    n_batches = len(test_loader)
    # test_set.reset_samples()

    with torch.no_grad():
        with tqdm(total=n_batches, desc='evaluating_cdisc', disable=True) as pbar:

            for batch in test_loader:

                x, y = batch['input'], batch['label']
                x = x.reshape(-1, *x.shape[2:])
                y = y.reshape(-1, *y.shape[2:])
                classification_weights = y * 100

                x = x.to(device=device, dtype=torch.float)
                y = y.to(device=device, dtype=torch.float)
                classification_weights = classification_weights.to(device=device, dtype=torch.float)

                predictions = student_model.forward(x)

                # loss = criterion(predictions, y)
                loss = criterion(predictions, y, classification_weights)
                running_loss += loss.item()
            
                probabilities = softmax(predictions)
                binary_prediction_vector = np.argmax(probabilities.cpu().numpy(), axis=1)
                target_vector = y[:, 1].cpu().numpy()

                correct_predictions = np.sum(binary_prediction_vector == target_vector)
                running_correct_predictions += correct_predictions
                total_samples += binary_prediction_vector.shape[0]

                accuracy = correct_predictions / binary_prediction_vector.shape[0]

                positives_predicted = np.sum(np.logical_and(binary_prediction_vector == 1, target_vector == 1))
                positives_sampled = np.sum(target_vector == 1)

                negatives_predicted = np.sum(np.logical_and(binary_prediction_vector == 0, target_vector == 0))
                negatives_sampled = np.sum(target_vector == 0)

                running_positive_sampled += positives_sampled
                running_positive_predictions += positives_predicted

                running_negative_sampled += negatives_sampled
                running_negative_predictions += negatives_predicted
            
                pbar.set_postfix({'loss': f'{loss.item():.6f}', 'accuracy': f'{accuracy:.6f}', 'positives predicted': f'({positives_predicted}/{positives_sampled})'})
                pbar.update(1)

    mean_validation_loss = (running_loss / n_batches)  # .cpu().numpy()
    mean_validation_accuracy = running_correct_predictions / total_samples
    print(f'Validation set average loss - {mean_validation_loss:.6f}, Average accuracy - {mean_validation_accuracy:.6f}, Positives predicted - ({running_positive_predictions}/{running_positive_sampled}), Negatives predicted - ({running_negative_predictions}/{running_negative_sampled})')

    return mean_validation_loss, mean_validation_accuracy