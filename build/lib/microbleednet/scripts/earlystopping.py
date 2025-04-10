from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import numpy as np

class EarlyStoppingModelCheckpointing:
    """
    Early stopping stops the training if the validation loss doesn't improve after a given patience
    """

    def __init__(self, model_name, patience=5, verbose=False):
        
        self.counter = 0
        self.best_score = None
        self.verbose = verbose
        self.early_stop = False
        self.model_name = model_name
        self.patience = patience
        self.best_validation_loss = np.inf

    def __call__(self, validation_loss, validation_dice, best_validation_dice, model, epoch, optimizer, scheduler, loss, training_params, weights=True, checkpoint=True, save_condition='best', model_path=None):
        
        score = -validation_loss
        # score = validation_dice
        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                print('Initial epoch;')
            self.save_checkpoint(validation_loss, validation_dice, best_validation_dice, model, epoch, optimizer, scheduler, loss, training_params, weights, checkpoint, save_condition, model_path)

        elif score < self.best_score:  # Here is the criteria for activation of early stopping counter.
            self.counter += 1
            print(f'Early Stopping Counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:  # When the counter reaches the patience value, early stopping flag is activated to stop the training.
                self.early_stop = True

        else:
            self.counter = 0
            self.best_score = score
            if self.verbose:
                print('Validation loss decreased;')
            self.save_checkpoint(validation_loss, validation_dice, best_validation_dice, model, epoch, optimizer, scheduler, loss, training_params, weights, checkpoint, save_condition, model_path)

    def save_checkpoint(self, validation_loss, validation_dice, best_validation_dice, model, epoch, optimizer, scheduler, loss, training_params, weights, checkpoint, save_condition, model_path):

        if checkpoint:

            if weights == True:
                if save_condition == 'best' and validation_dice > best_validation_dice:
                    torch.save(model.state_dict(), os.path.join(model_path, f'microbleednet_{self.model_name}_model_weights_bestdice.pth'))
                elif save_condition == 'everyN' and (epoch % training_params['EveryN']) == 0:
                    torch.save(model.state_dict(), os.path.join(model_path, f'microbleednet_{self.model_name}_model_weights_epoch_{epoch}.pth'))
                elif save_condition == 'last':
                    torch.save(model.state_dict(), os.path.join(model_path, f'microbleednet_{self.model_name}_model_weights.pth'))

                if self.verbose:
                    print('Saving model (only weights).')
                
            else:
                save_dict = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_stat_dict': scheduler.state_dict(),
                        'loss': loss
                    }
                
                if save_condition == 'best' and validation_dice > best_validation_dice:
                    torch.save(save_dict, os.path.join(model_path, f'microbleednet_{self.model_name}_model_bestdice.pth'))
                
                elif save_condition == 'everyN' and (epoch % training_params['EveryN']) == 0:
                    torch.save(save_dict, os.path.join(model_path, f'microbleednet_{self.model_name}_model_epoch_{epoch}.pth'))

                elif save_condition == 'last':
                    torch.save(save_dict, os.path.join(model_path, f'microbleednet_{self.model_name}_model_beforeES.pth'))

                if self.verbose:
                    print('Saving model (full environment).')
                
        elif self.verbose:
            print('Exiting without saving the model.')


class EarlyStoppingModelCheckpointing2models:
    """
    Early stopping stops the training if the validation loss doesnt improve after a given patience
    """

    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, val_dice, best_val_dice, model, model_class, epoch, optimizer, scheduler, loss,
                 tr_prms, weights=True, checkpoint=True, save_condition='best', model_path=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, val_dice, best_val_dice, model, epoch, optimizer, scheduler, loss,
                                 tr_prms, weights, checkpoint, save_condition, model_path)
        elif score < self.best_score:  # Here is the criteria for activation of early stopping counter.
            self.counter += 1
            print('Early Stopping Counter: ', self.counter, '/', self.patience)
            if self.counter >= self.patience:  # When the counter reaches the patience value, early stopping flag is activated to stop the training.
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, val_dice, best_val_dice, model, epoch, optimizer, scheduler, loss,
                                 tr_prms, weights, checkpoint, save_condition, model_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_acc, best_val_acc, model, model_class, epoch, optimizer, scheduler, loss,
                        tr_prms, weights, checkpoint, save_condition, PATH):
        # Saving checkpoints
        if checkpoint:
            # Saves the model when the validation loss decreases
            if self.verbose:
                print('Validation loss increased; Saving model ...')
            if weights:
                if save_condition == 'best':
                    save_path = os.path.join(PATH, 'Microbleednet_teacher_model_weights_bestdice.pth')
                    save_path_class = os.path.join(PATH, 'Microbleednet_teacher_model_class_weights_bestdice.pth')
                    if val_acc > best_val_acc:
                        torch.save(model.state_dict(), save_path)
                        torch.save(model_class.state_dict(), save_path_class)
                elif save_condition == 'everyN':
                    N = tr_prms['EveryN']
                    if (epoch % N) == 0:
                        save_path = os.path.join(PATH,
                                                 'Microbleednet_teacher_model_weights_epoch' + str(epoch) + '.pth')
                        save_path_class = os.path.join(PATH,
                                                       'Microbleednet_teacher_model_class_weights_epoch' + str(epoch) + '.pth')
                        torch.save(model.state_dict(), save_path)
                        torch.save(model_class.state_dict(), save_path_class)
                elif save_condition == 'last':
                    save_path = os.path.join(PATH, 'Microbleednet_teacher_model_weights_beforeES.pth')
                    save_path_class = os.path.join(PATH, 'Microbleednet_teacher_model_class_weights_beforeES.pth')
                    torch.save(model.state_dict(), save_path)
                    torch.save(model_class.state_dict(), save_path_class)
                else:
                    raise ValueError("Invalid saving condition provided! Valid options: best, everyN, last")
            else:
                if save_condition == 'best':
                    save_path = os.path.join(PATH, 'Microbleednet_teacher_model_bestdice.pth')
                    save_path_class = os.path.join(PATH, 'Microbleednet_teacher_model_class_bestdice.pth')
                    if val_acc > best_val_acc:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_stat_dict': scheduler.state_dict(),
                            'loss': loss
                        }, save_path)
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model_class.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_stat_dict': scheduler.state_dict(),
                            'loss': loss
                        }, save_path_class)
                elif save_condition == 'everyN':
                    N = tr_prms['EveryN']
                    if (epoch % N) == 0:
                        save_path = os.path.join(PATH, 'Microbleednet_teacher_model_epoch' + str(epoch) + '.pth')
                        save_path_class = os.path.join(PATH, 'Microbleednet_teacher_model_class_epoch' + str(epoch) + '.pth')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_stat_dict': scheduler.state_dict(),
                            'loss': loss
                        }, save_path)
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model_class.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_stat_dict': scheduler.state_dict(),
                            'loss': loss
                        }, save_path_class)
                elif save_condition == 'last':
                    save_path = os.path.join(PATH, 'Microbleednet_teacher_model_beforeES.pth')
                    save_path_class = os.path.join(PATH, 'Microbleednet_teacher_model_class_beforeES.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_stat_dict': scheduler.state_dict(),
                        'loss': loss
                    }, save_path)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_class.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_stat_dict': scheduler.state_dict(),
                        'loss': loss
                    }, save_path_class)
                else:
                    raise ValueError("Invalid saving condition provided! Valid options: best, everyN, last")
        else:
            if self.verbose:
                print('Validation loss increased; Exiting without saving the model ...')
