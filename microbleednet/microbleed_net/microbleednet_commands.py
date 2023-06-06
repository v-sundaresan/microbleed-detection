from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from microbleednet.microbleed_net import (microbleednet_cdet_train_function,
                                          microbleednet_cdisc_train_function,
                                          microbleednet_test_function,
                                          microbleednet_cdet_cross_validate, microbleednet_cdisc_cross_validate,
                                          microbleednet_cdet_finetune, microbleednet_cdisc_finetune)
import glob

#=========================================================================================
# microbleednet commands function
# Vaanathi Sundaresan
# 10-01-2023
#=========================================================================================


##########################################################################################
# Define the train sub-command for microbleednet
##########################################################################################

def train(args):
    '''
    :param args: Input arguments from argparse
    '''
    # Do basic sanity checks and assign variable names
    inp_dir = args.inp_dir

    if not os.path.isdir(inp_dir):
        raise ValueError(inp_dir + ' does not appear to be a valid input directory')

    input_paths = glob.glob(os.path.join(inp_dir, '*_preproc.nii')) + \
                  glob.glob(os.path.join(inp_dir, '*_preproc.nii.gz'))

    if len(input_paths) == 0:
        raise ValueError(inp_dir + ' does not contain any preprocessed input images / filenames NOT in required format')

    if os.path.isdir(args.model_dir) is False:
        raise ValueError(args.model_dir + ' does not appear to be a valid directory')
    model_dir = args.model_dir

    if os.path.isdir(args.label_dir) is False:
        raise ValueError(args.label_dir + ' does not appear to be a valid directory')
    label_dir = args.label_dir

    # Create a list of dictionaries containing required filepaths for the input subjects
    subj_name_dicts = []
    for l in range(len(input_paths)):
        basepath = input_paths[l].split("_preproc.nii")[0]
        basename = basepath.split(os.sep)[-1]

        print(os.path.join(label_dir, basename + '_manualmask.nii.gz'), flush=True)
        if os.path.isfile(os.path.join(label_dir, basename + '_manualmask.nii.gz')):
            gt_path_name = os.path.join(label_dir, basename + '_manualmask.nii.gz')
        elif os.path.isfile(os.path.join(label_dir, basename + '_manualmask.nii')):
            gt_path_name = os.path.join(label_dir, basename + '_manualmask.nii')
        else:
            raise ValueError('Manual lesion mask does not exist for ' + basename)

        subj_name_dict = {'flair_path': input_paths[l],
                          'gt_path': gt_path_name,
                          'basename': basename}
        subj_name_dicts.append(subj_name_dict)

    if isinstance(args.init_learng_rate, float) is False:
        raise ValueError('Initial learning rate must be a float value')
    else:
        if args.init_learng_rate > 1:
            raise ValueError('Initial learning rate must be between 0 and 1')

    if args.optimizer not in ['adam', 'sgd']:
        raise ValueError('Invalid option for Optimizer: Valid options: adam, sgd')

    if isinstance(args.lr_sch_gamma, float) is False:
        raise ValueError('Learning rate reduction factor must be a float value')
    else:
        if args.lr_sch_gamma > 1:
            raise ValueError('Learning rate reduction factor must be between 0 and 1')

    if isinstance(args.train_prop, float) is False:
        raise ValueError('Training data proportion must be a float value')
    else:
        if args.train_prop > 1:
            raise ValueError('Training data proportion must be between 0 and 1')

    if args.batch_size < 1:
        raise ValueError('Batch size must be an int and > 1')
    if args.num_epochs < 1:
        raise ValueError('Number of epochs must be an int and > 1')
    if args.batch_factor < 1:
        raise ValueError('Batch factor must be an int and > 1')
    if args.early_stop_val < 1 or args.early_stop_val > args.num_epochs:
        raise ValueError('Early stopping patience value must be an int and > 1 and < number of epochs')
    if args.aug_factor < 1:
        raise ValueError('Augmentation factor must be an int and > 1')
    if args.cp_save_type == 'everyN':
        if args.cp_everyn_N < 1 or args.cp_everyn_N > args.num_epochs:
            raise ValueError(
                'N value for saving checkpoints for every N epochs must be an int and > 1and < number of epochs')

    if args.num_classes < 1:
        raise ValueError('Number of classes to consider in target segmentations must be an int and > 1')

    # Create the training parameters dictionary
    training_params = {'Learning_rate': args.init_learng_rate,
                       'Optimizer': args.optimizer,
                       'Epsilon':args.epsilon,
                       'Momentum': args.momentum,
                       'LR_Milestones': args.lr_sch_mlstone,
                       'LR_red_factor': args.lr_sch_gamma,
                       'Train_prop': args.train_prop,
                       'Batch_size': args.batch_size,
                       'Num_epochs': args.num_epochs,
                       'Batch_factor': args.batch_factor,
                       'Patch_size': args.patch_size,
                       'Patience': args.early_stop_val,
                       'Aug_factor': args.aug_factor,
                       'EveryN': args.cp_everyn_N,
                       'Nclass': args.num_classes,
                       'SaveResume': args.save_resume_training
                       }

    if args.save_full_model == 'True':
        save_wei = False
    else:
        save_wei = True

    if args.cp_save_type not in ['best', 'last', 'everyN']:
        raise ValueError('Invalid option for checkpoint save type: Valid options: best, last, everyN')

    # Training main function call
    if args.cand_detection:
        models = microbleednet_cdet_train_function.main(subj_name_dicts, training_params, aug=args.data_augmentation,
                                                        save_cp=True, save_wei=save_wei, save_case=args.cp_save_type,
                                                        verbose=args.verbose, dir_cp=model_dir)

    if args.cand_discrimination:
        models = microbleednet_cdisc_train_function.main(subj_name_dicts, training_params, aug=args.data_augmentation,
                                                         save_cp=True, save_wei=save_wei, save_case=args.cp_save_type,
                                                         verbose=args.verbose, dir_cp=model_dir)


##########################################################################################
# Define the evaluate sub-command for microbleednet
##########################################################################################

def evaluate(args):
    '''
    :param args: Input arguments from argparse
    '''
    # Do basic sanity checks and assign variable names
    inp_dir = args.inp_dir
    out_dir = args.output_dir

    if not os.path.isdir(inp_dir):
        raise ValueError(inp_dir + ' does not appear to be a valid input directory')

    input_paths = glob.glob(os.path.join(inp_dir, '*_preproc.nii')) + \
                  glob.glob(os.path.join(inp_dir, '*_preproc.nii.gz'))

    if len(input_paths) == 0:
        raise ValueError(inp_dir + ' does not contain any preprocessed input images / filenames NOT in required format')

    if os.path.isdir(out_dir) is False:
        raise ValueError(out_dir + ' does not appear to be a valid directory')

    # Create a list of dictionaries containing required filepaths for the test subjects
    subj_name_dicts = []
    for l in range(len(input_paths)):
        basepath = input_paths[l].split("_preproc.nii")[0]
        basename = basepath.split(os.sep)[-1]

        subj_name_dict = {'flair_path': input_paths[l],
                          'basename': basename}
        subj_name_dicts.append(subj_name_dict)

    if args.num_classes < 1:
        raise ValueError('Number of classes to consider in target segmentations must be an int and > 1')

    if args.pretrained_model == 'True':
        model_name = None
        model_dir = os.path.expandvars('$FSLDIR/data/microbleednet/models')
        if not os.path.exists(model_dir):
            model_dir = os.environ.get('microbleednet_PRETRAINED_MODEL_PATH', None)
            if model_dir is None:
                raise RuntimeError('Cannot find data; export microbleednet_PRETRAINED_MODEL_PATH=/path/to/my/mwsc/model')
    else:
        if os.path.isfile(args.model_name + '_cdet.pth') is False or \
                os.path.isfile(args.model_name + '_cdisc_student.pth') is False:
            raise ValueError('In directory ' + os.path.dirname(args.model_name) +
                             ', ' + os.path.basename(args.model_name) + '_cdet.pth or' +
                             os.path.basename(args.model_name) + '_cdisc_student.pth' +
                             'does not appear to be a valid model file')
        else:
            model_dir = os.path.dirname(args.model_name)
            model_name = os.path.basename(args.model_name)

    # Create the training parameters dictionary
    eval_params = {'Nclass': args.num_classes,
                   'EveryN': args.cp_everyn_N,
                   'Pretrained': args.pretrained_model,
                   'Modelname': model_name
                   }

    if args.cp_load_type not in ['best', 'last', 'specific']:
        raise ValueError('Invalid option for checkpoint save type: Valid options: best, last, specific')

    if args.cp_load_type == 'specific':
        args.cp_load_type = 'everyN'
        if args.cp_everyn_N is None:
            raise ValueError('-cp_n must be provided to specify the epoch for loading CP when using -cp_type is "specific"!')

    # Test main function call
    microbleednet_test_function.main(subj_name_dicts, eval_params, intermediate=args.intermediate,
                                     model_dir=model_dir, load_case=args.cp_load_type, output_dir=out_dir,
                                     verbose=args.verbose)


##########################################################################################
# Define the fine_tune sub-command for microbleednet
##########################################################################################

def fine_tune(args):
    '''
    :param args: Input arguments from argparse
    '''
    # Do the usual sanity checks
    inp_dir = args.inp_dir

    if not os.path.isdir(inp_dir):
        raise ValueError(inp_dir + ' does not appear to be a valid input directory')

    input_paths = glob.glob(os.path.join(inp_dir, '*_preproc.nii')) + \
                  glob.glob(os.path.join(inp_dir, '*_preproc.nii.gz'))

    if len(input_paths) == 0:
        raise ValueError(inp_dir + ' does not contain any preprocessed input images / filenames NOT in required format')

    out_dir = args.output_dir
    if os.path.isdir(out_dir) is False:
        raise ValueError(out_dir + ' does not appear to be a valid directory')

    if os.path.isdir(args.label_dir) is False:
        raise ValueError(args.label_dir + ' does not appear to be a valid directory')
    label_dir = args.label_dir

    # Create a list of dictionaries containing required filepaths for the fine-tuning subjects
    subj_name_dicts = []
    for l in range(len(input_paths)):
        basepath = input_paths[l].split("_preproc.nii")[0]
        basename = basepath.split(os.sep)[-1]

        if os.path.isfile(os.path.join(label_dir, basename + '_manualmask.nii.gz')):
            gt_path_name = os.path.join(label_dir, basename + '_manualmask.nii.gz')
        elif os.path.isfile(os.path.join(label_dir, basename + '_manualmask.nii')):
            gt_path_name = os.path.join(label_dir, basename + '_manualmask.nii')
        else:
            raise ValueError('Manual lesion mask does not exist for ' + basename)

        subj_name_dict = {'flair_path': input_paths[l],
                          'gt_path': gt_path_name,
                          'basename': basename}
        subj_name_dicts.append(subj_name_dict)

    if isinstance(args.init_learng_rate, float) is False:
        raise ValueError('Initial learning rate must be a float value')
    else:
        if args.init_learng_rate > 1:
            raise ValueError('Initial learning rate must be between 0 and 1')

    if args.optimizer not in ['adam', 'sgd']:
        raise ValueError('Invalid option for Optimizer: Valid options: adam, sgd')

    if isinstance(args.lr_sch_gamma, float) is False:
        raise ValueError('Learning rate reduction factor must be a float value')
    else:
        if args.lr_sch_gamma > 1:
            raise ValueError('Learning rate reduction factor must be between 0 and 1')

    if isinstance(args.train_prop, float) is False:
        raise ValueError('Training data proportion must be a float value')
    else:
        if args.train_prop > 1:
            raise ValueError('Training data proportion must be between 0 and 1')

    if args.batch_size < 1:
        raise ValueError('Batch size must be an int and > 1')
    if args.num_epochs < 1:
        raise ValueError('Number of epochs must be an int and > 1')
    if args.batch_factor < 1:
        raise ValueError('Batch factor must be an int and > 1')
    if args.early_stop_val < 1 or args.early_stop_val > args.num_epochs:
        raise ValueError('Early stopping patience value must be an int and > 1 and < number of epochs')
    if args.aug_factor < 1:
        raise ValueError('Augmentation factor must be an int and > 1')

    if args.cp_save_type == 'everyN':
        if args.cp_everyn_N < 1 or args.cp_everyn_N > args.num_epochs:
            raise ValueError(
                'N value for saving checkpoints for every N epochs must be an int and > 1and < number of epochs')
    if args.num_classes < 1:
        raise ValueError('Number of classes to consider in target segmentations must be an int and > 1')

    if args.save_full_model == 'True':
        save_wei = False
    else:
        save_wei = True

    if args.pretrained_model == 'True':
        model_name = None
        model_dir = os.path.expandvars('$FSLDIR/data/microbleednet/models')
        if not os.path.exists(model_dir):
            model_dir = os.environ.get('microbleednet_PRETRAINED_MODEL_PATH', None)
            if model_dir is None:
                raise RuntimeError(
                    'Cannot find data; export microbleednet_PRETRAINED_MODEL_PATH=/path/to/my/mwsc/model')
    else:
        if os.path.isfile(args.model_name + '_cdet.pth') is False or \
                os.path.isfile(args.model_name + '_cdisc_student.pth') is False:
            raise ValueError('In directory ' + os.path.dirname(args.model_name) +
                             ', ' + os.path.basename(args.model_name) + '_cdet.pth or' +
                             os.path.basename(args.model_name) + '_cdisc_student.pth' +
                             'does not appear to be a valid model file')
        else:
            model_dir = os.path.dirname(args.model_name)
            model_name = os.path.basename(args.model_name)

    # Create the fine-tuning parameters dictionary
    finetuning_params = {'Finetuning_learning_rate': args.init_learng_rate,
                         'Optimizer': args.optimizer,
                         'Epsilon': args.epsilon,
                         'Momentum': args.momentum,
                         'LR_Milestones': args.lr_sch_mlstone,
                         'LR_red_factor': args.lr_sch_gamma,
                         'Train_prop': args.train_prop,
                         'Batch_size': args.batch_size,
                         'Num_epochs': args.num_epochs,
                         'Batch_factor': args.batch_factor,
                         'Patch_size':args.patch_size,
                         'Patience': args.early_stop_val,
                         'Aug_factor': args.aug_factor,
                         'EveryN': args.cp_everyn_N,
                         'Nclass': args.num_classes,
                         'Finetuning_layers': args.ft_layers,
                         'Load_type': args.cp_load_type,
                         'EveryNload': args.cpload_everyn_N,
                         'Pretrained': args.pretrained_model,
                         'Modelname': model_name,
                         'SaveResume': args.save_resume_training
                         }

    if args.cp_save_type not in ['best', 'last', 'everyN']:
        raise ValueError('Invalid option for checkpoint save type: Valid options: best, last, everyN')

    if args.cp_save_type == 'everyN':
        if args.cp_everyn_N is None:
            raise ValueError('-cp_n must be provided to specify the epoch for loading CP when using -cp_type is "everyN"!')

    # Fine-tuning main function call
    if args.cand_detection:
        microbleednet_cdet_finetune.main(subj_name_dicts, finetuning_params, aug=args.data_augmentation, weighted=weighted,
                          save_cp=True, save_wei=save_wei, save_case=args.cp_save_type, verbose=args.verbose,
                          model_dir=model_dir, dir_cp=out_dir)

    if args.cand_discrimination:
        microbleednet_cdisc_finetune.main(subj_name_dicts, finetuning_params, aug=args.data_augmentation, weighted=weighted,
                          save_cp=True, save_wei=save_wei, save_case=args.cp_save_type, verbose=args.verbose,
                          model_dir=model_dir, dir_cp=out_dir)

##########################################################################################
# Define the loo_validate (leave-one-out validation) sub-command for microbleednet
##########################################################################################

def cross_validate(args):
    '''
    :param args: Input arguments from argparse
    '''
    # Usual sanity check for checking if filepaths and files exist.
    inp_dir = args.inp_dir

    if not os.path.isdir(inp_dir):
        raise ValueError(inp_dir + ' does not appear to be a valid input directory')

    input_paths = glob.glob(os.path.join(inp_dir, '*_preproc.nii')) + \
                  glob.glob(os.path.join(inp_dir, '*_preproc.nii.gz'))

    if len(input_paths) == 0:
        raise ValueError(inp_dir + ' does not contain any preprocessed input images / filenames NOT in required format')

    if os.path.isdir(args.output_dir) is False:
        raise ValueError(args.output_dir + ' does not appear to be a valid directory')
    out_dir = args.output_dir

    if os.path.isdir(args.label_dir) is False:
        raise ValueError(args.label_dir + ' does not appear to be a valid directory')
    label_dir = args.label_dir

    # if os.path.isdir(model_dir) is False:
    #     raise ValueError(model_dir + ' does not appear to be a valid directory')

    if args.cv_fold < 1:
        raise ValueError('Number of folds cannot be 0 or negative')

    if args.resume_from_fold < 1:
        raise ValueError('Fold to resume cannot be 0 or negative')

    # Create a list of dictionaries containing required filepaths for the input subjects
    subj_name_dicts = []
    for l in range(len(input_paths)):
        basepath = input_paths[l].split("_FLAIR.nii")[0]
        basename = basepath.split(os.sep)[-1]

        if os.path.isfile(os.path.join(label_dir, basename + '_manualmask.nii.gz')):
            gt_path_name = os.path.join(label_dir, basename + '_manualmask.nii.gz')
        elif os.path.isfile(os.path.join(label_dir, basename + '_manualmask.nii')):
            gt_path_name = os.path.join(label_dir, basename + '_manualmask.nii')
        else:
            raise ValueError('Manual lesion mask does not exist for ' + basename)

        subj_name_dict = {'flair_path': input_paths[l],
                          'gt_path': gt_path_name,
                          'basename': basename}
        subj_name_dicts.append(subj_name_dict)

    if isinstance(args.init_learng_rate, float) is False:
        raise ValueError('Initial learning rate must be a float value')
    else:
        if args.init_learng_rate > 1:
            raise ValueError('Initial learning rate must be between 0 and 1')

    if args.optimizer not in ['adam', 'sgd']:
        raise ValueError('Invalid option for Optimizer: Valid options: adam, sgd')

    if isinstance(args.lr_sch_gamma, float) is False:
        raise ValueError('Learning rate reduction factor must be a float value')
    else:
        if args.lr_sch_gamma > 1:
            raise ValueError('Learning rate reduction factor must be between 0 and 1')

    if isinstance(args.train_prop, float) is False:
        raise ValueError('Training data proportion must be a float value')
    else:
        if args.train_prop > 1:
            raise ValueError('Training data proportion must be between 0 and 1')

    if args.batch_size < 1:
        raise ValueError('Batch size must be an int and > 1')
    if args.num_epochs < 1:
        raise ValueError('Number of epochs must be an int and > 1')
    if args.batch_factor < 1:
        raise ValueError('Batch factor must be an int and > 1')
    if args.early_stop_val < 1 or args.early_stop_val > args.num_epochs:
        raise ValueError('Early stopping patience value must be an int and > 1 and < number of epochs')
    if args.aug_factor < 1:
        raise ValueError('Augmentation factor must be an int and > 1')
    # if args.cp_save_type == 'everyN':
    #     if args.cp_everyn_N < 1 or args.cp_everyn_N > args.num_epochs:
    #         raise ValueError(
    #             'N value for saving checkpoints for every N epochs must be an int and > 1and < number of epochs')

    if args.num_classes < 1:
        raise ValueError('Number of classes to consider in target segmentations must be an int and > 1')

    if len(subj_name_dicts) < args.cv_fold:
        raise ValueError('Number of folds is greater than number of subjects!')

    if args.resume_from_fold > args.cv_fold:
        raise ValueError('The fold to resume CV cannot be higher than the total number of folds specified!')

    # Create the loo_validate parameters dictionary
    cv_params = {'Learning_rate': args.init_learng_rate,
                 'fold': args.cv_fold,
                 'res_fold': args.resume_from_fold,
                 'Optimizer': args.optimizer,
                 'Epsilon': args.epsilon,
                 'Momentum': args.momentum,
                 'LR_Milestones': args.lr_sch_mlstone,
                 'LR_red_factor': args.lr_sch_gamma,
                 'Train_prop': args.train_prop,
                 'Batch_size': args.batch_size,
                 'Patch_size': args.patch_size,
                 'Num_epochs': args.num_epochs,
                 'Batch_factor': args.batch_factor,
                 'Patience': args.early_stop_val,
                 'Aug_factor': args.aug_factor,
                 'Nclass': args.num_classes,
                 'EveryN': args.cp_everyn_N,
                 'SaveResume': args.save_resume_training
                 }

    if args.save_full_model == 'True':
        save_wei = False
    else:
        save_wei = True

    if args.cp_save_type not in ['best', 'last', 'everyN']:
        raise ValueError('Invalid option for checkpoint save type: Valid options: best, last, everyN')

    if args.cp_save_type == 'everyN':
        if args.cp_everyn_N is None:
            raise ValueError('-cp_n must be provided to specify the epoch for loading CP when using -cp_type is "everyN"!')

    # Cross-validation main function call
    if args.cand_detection:
        microbleednet_cdet_cross_validate.main(subj_name_dicts, cv_params, aug=args.data_augmentation,
                                               intermediate=args.intermediate, save_cp=args.save_checkpoint,
                                               save_wei=save_wei, save_case=args.cp_save_type, verbose=args.verbose,
                                               dir_cp=out_dir, output_dir=out_dir)

    if args.cand_discrimination:
        microbleednet_cdisc_cross_validate.main(subj_name_dicts, cv_params, aug=args.data_augmentation,
                                                intermediate=args.intermediate, save_cp=args.save_checkpoint,
                                                save_wei=save_wei, save_case=args.cp_save_type, verbose=args.verbose,
                                                dir_cp=out_dir, output_dir=out_dir)

