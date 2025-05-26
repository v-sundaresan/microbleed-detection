from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import subprocess
import nibabel as nib
from glob import glob
from tqdm import tqdm

from microbleednet.scripts import data_preparation
from microbleednet.scripts import evaluate_function
from microbleednet.scripts import cdet_train_function
from microbleednet.scripts import cdisc_train_function
from microbleednet.scripts import crossvalidate_function
from microbleednet.scripts import cdet_finetune_function
from microbleednet.scripts import cdisc_finetune_function

###################################
# microbleednet commands function #
# Vaanathi Sundaresan             #
# 10-01-2023                      #
###################################

############################################
# Preprocess sub-command for microbleednet #
############################################

def preprocess(args):
    """
    :param args: Input arguments from argparse
    """

    input_directory = args.inp_dir
    output_directory = args.out_dir
    label_directory = args.label_dir
    input_file_regex = args.image_regex
    label_file_regex = args.label_regex
    fsl_preprocessed = args.fsl_preprocessed

    # Check if input directory is valid
    if not os.path.isdir(input_directory):
        raise ValueError(f'{input_directory} does not appear to be a valid input directory')
    
    input_paths = glob(os.path.join(input_directory, '**', f'*{input_file_regex}.nii*'), recursive=True)
    if len(input_paths) == 0:
        raise ValueError(f'{input_directory} does not contain any input images / filenames NOT in required format')
    
    # Check if label directory is valid (and if it exists)
    if label_directory is not None and os.path.isdir(label_directory) is False:
        raise ValueError(f'{label_directory} does not appear to be a valid directory')

    for input_path in tqdm(input_paths, leave=False, desc='Preprocessing subjects', disable=True):

        basepath = input_path.split(input_file_regex)[0]
        basename = basepath.split(os.sep)[-1]

        if not fsl_preprocessed:

            subdirectory = basepath.split('/')[-2]
            basename = subdirectory +'-' + basename
            
            os.makedirs(os.path.join(output_directory, 'fsl_preprocessed', 'images'), exist_ok=True)
            fsl_image_output_path = os.path.join(output_directory, 'fsl_preprocessed', 'images', basename)
            subprocess.run(['bash', 'skull_strip_bias_field_correct.sh', input_path, fsl_image_output_path], check=True)

            input_path = os.path.join(output_directory, 'fsl_preprocessed', 'images', basename + '_preproc.nii.gz')

            if label_directory is not None:
            
                label_extensions = label_file_regex
                for extension in label_extensions:
                    label_path = basepath + extension
                    if os.path.isfile(label_path):
                        break
                else:
                    raise ValueError(f'Manual lesion mask does not exist for {basename}, {label_path}')

                os.makedirs(os.path.join(output_directory, 'fsl_preprocessed', 'labels'), exist_ok=True)
                label_output_path = os.path.join(output_directory, 'fsl_preprocessed', 'labels', basename + '_mask_preproc.nii.gz')
                subprocess.run(["cp", label_path, label_output_path])

                label_path = label_output_path

        elif label_directory is not None:
            # Checks if the manual label exists for the current file
            label_extensions = label_file_regex + ['_mask_preproc.nii.gz']
            for extension in label_extensions:
                label_path = os.path.join(label_directory, basename + extension)
                if os.path.isfile(label_path):
                    break
            else:
                raise ValueError(f'Manual lesion mask does not exist for {basename}, {basename + extension}')
        
        subject = {
            'basename': basename,
            'input_path': input_path,
        }

        if label_directory is not None:
            subject['label_path'] = label_path

        image, label, frst = data_preparation.preprocess_subject(subject)

        header = nib.load(input_path).header
        affine = nib.load(input_path).affine

        os.makedirs(os.path.join(output_directory, 'images'), exist_ok=True)
        image_path = os.path.join(output_directory, 'images', basename + '_preproc.nii.gz')
        obj = nib.nifti1.Nifti1Image(image, affine, header=header)
        nib.save(obj, image_path)

        if label_directory is not None:
            os.makedirs(os.path.join(output_directory, 'labels'), exist_ok=True)
            label_path = os.path.join(output_directory, 'labels', basename + '_mask.nii.gz')
            obj = nib.nifti1.Nifti1Image(label, affine, header=header)
            nib.save(obj, label_path)

        os.makedirs(os.path.join(output_directory, 'frsts'), exist_ok=True)
        frst_path = os.path.join(output_directory, 'frsts', basename + '_frst.nii.gz')
        obj = nib.nifti1.Nifti1Image(frst, affine, header=header)
        nib.save(obj, frst_path)

    print('All subjects preprocessed.')


#######################################
# Train sub-command for microbleednet #
#######################################

def train(args):
    """
    :param args: Input arguments from argparse
    """

    preprocessed_directory = args.inp_dir
    # label_directory = args.label_dir
    model_directory = args.model_dir

    # Check if preprocessed directory is valid
    if not os.path.isdir(preprocessed_directory):
        raise ValueError(f'{preprocessed_directory} does not appear to be a valid input directory')
    
    input_directory = os.path.join(preprocessed_directory, 'images')
    label_directory = os.path.join(preprocessed_directory, 'labels')
    frst_directory = os.path.join(preprocessed_directory, 'frsts')

    input_paths = glob(os.path.join(input_directory, '*_preproc.nii*'))

    # Check if input directory actually contains files
    if len(input_paths) == 0:
        raise ValueError(f'{input_directory} does not contain any preprocessed input images / filenames NOT in required format')

    # Check if label directory is valid
    if os.path.isdir(label_directory) is False:
        raise ValueError(f'{label_directory} does not appear to be a valid directory, please preprocess images')
    
    # Check if FRST directory is valid
    if os.path.isdir(frst_directory) is False:
        raise ValueError(f'{frst_directory} does not appear to be a valid directory, please preprocess images')
    
    # Check if model directory is valid
    if os.path.isdir(model_directory) is False:
        raise ValueError(f'{model_directory} does not appear to be a valid directory')

    # Create a list of dictionaries containing required filepaths for the input subjects
    subjects = []
    for input_path in input_paths:

        basepath = input_path.split('_preproc.nii')[0]
        basename = basepath.split(os.sep)[-1]
        
        # Checks if the manual label exists for the current file
        label_path = os.path.join(label_directory, basename + '_mask.nii.gz')
        if not os.path.isfile(label_path):
            raise ValueError(f'Manual lesion mask does not exist for {basename}')
        
        # Checks if the FRST exists for the current file
        frst_path = os.path.join(frst_directory, basename + '_frst.nii.gz')
        if not os.path.isfile(frst_path):
            raise ValueError(f'FRST does not exist for {basename}')
        
        subject = {
            'basename': basename,
            'input_path': input_path,
            'label_path': label_path,
            'frst_path': frst_path,
        }

        subjects.append(subject)

    if isinstance(args.init_learng_rate, float) is False:
        raise ValueError('Initial learning rate must be a float value.')
    elif args.init_learng_rate > 1:
        raise ValueError('Initial learning rate must be between 0 and 1.')

    if args.optimizer not in ['adam', 'sgd']:
        raise ValueError('Invalid option for Optimizer. Valid options are: adam, sgd.')

    if isinstance(args.lr_sch_gamma, float) is False:
        raise ValueError('Learning rate reduction factor must be a float value.')
    elif args.lr_sch_gamma > 1:
        raise ValueError('Learning rate reduction factor must be between 0 and 1.')

    if isinstance(args.train_prop, float) is False:
        raise ValueError('Training data proportion must be a float value.')
    elif args.train_prop > 1:
        raise ValueError('Training data proportion must be between 0 and 1.')

    if args.batch_size < 1:
        raise ValueError('Batch size must be an int and > 1.')
    if args.num_epochs < 1:
        raise ValueError('Number of epochs must be an int and > 1.')
    if args.batch_factor < 1:
        raise ValueError('Batch factor must be an int and > 1.')
    if args.early_stop_val < 1 or args.early_stop_val > args.num_epochs:
        raise ValueError('Early stopping patience value must be an int and > 1 and < number of epochs.')
    if args.aug_factor < 1:
        raise ValueError('Augmentation factor must be an int and > 1.')
    if args.cp_save_type not in ['best', 'last', 'everyN']:
        raise ValueError('Invalid option for checkpoint save type. Valid options are: best, last, everyN')
    if args.cp_save_type == 'everyN':
        if args.cp_everyn_N is None:
            raise ValueError('-cp_n must be provided to specify the epoch for loading CP when using -cp_type is "everyN"!')
        if args.cp_everyn_N < 1 or args.cp_everyn_N > args.num_epochs:
            raise ValueError('N value for saving checkpoints for every N epochs must be an int and > 1and < number of epochs.')
        
    # This arg does not exist in the argparser
    # if args.num_classes < 1:
    #     raise ValueError('Number of classes to consider in target segmentations must be an int and > 1.')

    # Inverts the value stored in args.save_full_model and converts it to bool (this is from original code)
    save_weights = args.save_full_model != 'True'

    # Create the training parameters dictionary
    training_params = {
        'Learning_rate': args.init_learng_rate,
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
        'SaveResume': args.save_resume_training
    }

    if args.verbose:
        parameters = vars(args)
        print('Input parameters are:')
        for k, v in parameters.items():
            print(f'{k:<25} {v}')
        print()

    # This seems unnecessary if we're taking model_directory as a command-line argument from the user

    # model_directory = os.path.expandvars('$FSLDIR/data/microbleednet/models')
    # if not os.path.exists(model_directory):
    #     model_directory = os.environ.get('MICROBLEEDNET_PRETRAINED_MODEL_PATH', None)
    #     if model_directory is None:
    #         raise RuntimeError('Please export MICROBLEEDNET_PRETRAINED_MODEL_PATH=/path/to/your/model')

    # Call main training functions
    if args.cand_detection:
        models = cdet_train_function.main(subjects, training_params, perform_augmentation=args.data_augmentation, save_checkpoint=True, save_weights=save_weights, save_case=args.cp_save_type, verbose=args.verbose, checkpoint_directory=model_directory)

    if args.verbose:
        print('Trained CDet.')

    if args.cand_discrimination:
        models = cdisc_train_function.main(subjects, training_params, model_directory=model_directory, perform_augmentation=args.data_augmentation, save_checkpoint=True, save_weights=save_weights, save_case=args.cp_save_type, verbose=args.verbose, checkpoint_directory=model_directory)

    if args.verbose:
        print('Trained CDisc')
        print('Training complete!')


#####################################################
# Define the evaluate sub-command for microbleednet #
#####################################################

def evaluate(args):
    """
    :param args: Input arguments from argparse
    """

    preprocessed_directory = args.inp_dir
    output_directory = args.output_dir

    # Check if input directory is valid
    if not os.path.isdir(preprocessed_directory):
        raise ValueError(f'{preprocessed_directory} does not appear to be a valid input directory')
    
    input_directory = os.path.join(preprocessed_directory, 'images')
    frst_directory = os.path.join(preprocessed_directory, 'frsts')

    input_paths = glob(os.path.join(input_directory, '*_preproc.nii*'))

    # Check if input directory actually contains files
    if len(input_paths) == 0:
        raise ValueError(f'{input_directory} does not contain any preprocessed input images / filenames NOT in required format')

    # Check if FRST directory is valid
    if os.path.isdir(frst_directory) is False:
        raise ValueError(f'{frst_directory} does not appear to be a valid directory, please preprocess images')

    # Check if output directory is valid
    if os.path.isdir(output_directory) is False:
        raise ValueError(f'{output_directory} does not appear to be a valid directory')

    # Create a list of dictionaries containing required filepaths for the input subjects
    subjects = []
    for input_path in input_paths:

        basepath = input_path.split('_preproc.nii')[0]
        basename = basepath.split(os.sep)[-1]
        
        # Checks if the FRST exists for the current file
        frst_path = os.path.join(frst_directory, basename + '_frst.nii.gz')
        if not os.path.isfile(frst_path):
            raise ValueError(f'FRST does not exist for {basename}')
        
        subject = {
            'basename': basename,
            'input_path': input_path,
            'frst_path': frst_path,
        }

        subjects.append(subject)

    if args.model_name == 'pre':
        model_name = 'microbleednet'
        model_directory = os.path.expandvars('$FSLDIR/data/microbleednet/models')

        if not os.path.exists(model_directory):
            model_directory = os.environ.get('MICROBLEEDNET_PRETRAINED_MODEL_PATH')

            if model_directory is None:
                raise RuntimeError('Cannot find data; export MICROBLEEDNET_PRETRAINED_MODEL_PATH=/path/to/my/model')
            
    else:
        # Check if model paths are valid
        if not os.path.isfile(f'{args.model_name}_cdet_model_weights.pth'):
            raise ValueError(f'In directory {os.path.dirname(args.model_name)}, {os.path.basename(args.model_name)}_cdet_model.pth does not appear to be a valid file.')
        if not os.path.isfile(f'{args.model_name}_cdisc_student_model_weights.pth'):
            raise ValueError(f'In directory {os.path.dirname(args.model_name)}, {os.path.basename(args.model_name)}_cdisc_student_model.pth does not appear to be a valid file.')

        model_name = os.path.basename(args.model_name)
        model_directory = os.path.dirname(args.model_name)

    # Create the evaluation parameters dictionary
    evaluation_parameters = {
        'EveryN': args.cp_everyn_N,
        'Modelname': model_name,
    }

    if args.verbose:
        parameters = vars(args)
        print('Input parameters are:')
        for k, v in parameters.items():
            print(f'{k:<25} {v}')
        print()

    if args.cp_load_type not in ['best', 'last', 'specific']:
        raise ValueError('Invalid option for checkpoint save type: Valid options: best, last, specific')

    if args.cp_load_type == 'specific':
        args.cp_load_type = 'everyN'
        if args.cp_everyn_N is None:
            raise ValueError('-cp_n must be provided to specify the epoch for loading CP when using -cp_type is "specific"!')

    # Call the evaluate function
    evaluate_function.main(subjects, evaluation_parameters, args.intermediate, model_directory, args.cp_load_type, output_directory, args.verbose)


######################################################
# Define the fine_tune sub-command for microbleednet #
######################################################

def fine_tune(args):
    '''
    :param args: Input arguments from argparse
    '''
    # Do the usual sanity checks
    preprocessed_directory = args.inp_dir
    # label_directory = args.label_dir
    output_directory = args.output_dir

    if not os.path.isdir(preprocessed_directory):
        raise ValueError(f'{preprocessed_directory} does not appear to be a valid input directory')
    
    input_directory = os.path.join(preprocessed_directory, 'images')
    label_directory = os.path.join(preprocessed_directory, 'labels')
    frst_directory = os.path.join(preprocessed_directory, 'frsts')

    input_paths = glob(os.path.join(input_directory, '*_preproc.nii*'))

    # Check if input directory actually contains files
    if len(input_paths) == 0:
        raise ValueError(f'{input_directory} does not contain any preprocessed input images / filenames NOT in required format')
    
    # Check if label directory is valid
    if os.path.isdir(args.label_dir) is False:
        raise ValueError(args.label_dir + f'{label_directory} does not appear to be a valid directory')
    
    # Check if FRST directory is valid
    if os.path.isdir(frst_directory) is False:
        raise ValueError(f'{frst_directory} does not appear to be a valid directory, please preprocess images')

    # Check if output directory is valid
    if os.path.isdir(output_directory) is False:
        raise ValueError(f'{output_directory} does not appear to be a valid directory')

    # Create a list of dictionaries containing required filepaths for the fine-tuning subjects
    subjects = []
    for input_path in input_paths:

        basepath = input_path.split('_preproc.nii')[0]
        basename = basepath.split(os.sep)[-1]

        # Checks if the manual label exists for the current file
        label_path = os.path.join(label_directory, basename + '_mask.nii.gz')
        if not os.path.isfile(label_path):
            raise ValueError(f'Manual lesion mask does not exist for {basename}')
        
        # Checks if the FRST exists for the current file
        frst_path = os.path.join(frst_directory, basename + '_frst.nii.gz')
        if not os.path.isfile(frst_path):
            raise ValueError(f'FRST does not exist for {basename}')

        subject = {
            'basename': basename,
            'input_path': input_path,
            'label_path': label_path,
            'frst_path': frst_path,
        }

        subjects.append(subject)

    if isinstance(args.init_learng_rate, float) is False:
        raise ValueError('Initial learning rate must be a float value')
    elif args.init_learng_rate > 1:
        raise ValueError('Initial learning rate must be between 0 and 1')

    if args.optimizer not in ['adam', 'sgd']:
        raise ValueError('Invalid option for Optimizer. Valid options are: adam, sgd.')

    if isinstance(args.lr_sch_gamma, float) is False:
        raise ValueError('Learning rate reduction factor must be a float value.')
    elif args.lr_sch_gamma > 1:
        raise ValueError('Learning rate reduction factor must be between 0 and 1.')

    if isinstance(args.train_prop, float) is False:
        raise ValueError('Training data proportion must be a float value.')
    elif args.train_prop > 1:
        raise ValueError('Training data proportion must be between 0 and 1.')

    if args.batch_size < 1:
        raise ValueError('Batch size must be an int and > 1.')
    if args.num_epochs < 1:
        raise ValueError('Number of epochs must be an int and > 1.')
    if args.batch_factor < 1:
        raise ValueError('Batch factor must be an int and > 1.')
    if args.early_stop_val < 1 or args.early_stop_val > args.num_epochs:
        raise ValueError('Early stopping patience value must be an int and > 1 and < number of epochs.')
    if args.aug_factor < 1:
        raise ValueError('Augmentation factor must be an int and > 1.')
    if args.cp_save_type not in ['best', 'last', 'everyN']:
        raise ValueError('Invalid option for checkpoint save type. Valid options are: best, last, everyN')
    if args.cp_save_type == 'everyN':
        if args.cp_everyn_N is None:
            raise ValueError('-cp_n must be provided to specify the epoch for loading CP when using -cp_type is "everyN"!')
        if args.cp_everyn_N < 1 or args.cp_everyn_N > args.num_epochs:
            raise ValueError(
                'N value for saving checkpoints for every N epochs must be an int and > 1and < number of epochs')
        
    # This arg does not exist in the argparser
    # if args.num_classes < 1:
    #     raise ValueError('Number of classes to consider in target segmentations must be an int and > 1')

    # Inverts the value stored in args.save_full_model and converts it to bool (this is from original code)
    save_weights = args.save_full_model != 'True'

    if args.model_name == 'pre':
        model_name = 'microbleednet'
        model_directory = os.path.expandvars('$FSLDIR/data/microbleednet/models')

        if not os.path.exists(model_directory):
            model_directory = os.environ.get('MICROBLEEDNET_PRETRAINED_MODEL_PATH', None)

            if model_directory is None:
                raise RuntimeError(
                    'Cannot find data; export MICROBLEEDNET_PRETRAINED_MODEL_PATH=/path/to/my/mwsc/model')
            
    else:
        # Check if model paths are valid
        if not os.path.isfile(f'{args.model_name}_cdet_model_weights.pth'):
            raise ValueError(f'In directory {os.path.dirname(args.model_name)}, {os.path.basename(args.model_name)}_cdet_model_weights.pth does not appear to be a valid file.')
        if not os.path.isfile(f'{args.model_name}_cdisc_student_model_weights.pth'):
            raise ValueError(f'In directory {os.path.dirname(args.model_name)}, {os.path.basename(args.model_name)}_cdisc_student_model_weights.pth does not appear to be a valid file.')
        
        model_name = os.path.basename(args.model_name)
        model_directory = os.path.dirname(args.model_name)

    # Create the fine-tuning parameters dictionary
    finetune_params = {
        'Finetuning_learning_rate': args.init_learng_rate,
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
        'Finetuning_layers': args.ft_layers,
        'Load_type': args.cp_load_type,
        'EveryNload': args.cpload_everyn_N,
        'Modelname': model_name,
        'SaveResume': args.save_resume_training,
        }
    
    if args.verbose:
        parameters = vars(args)
        print('Input parameters are:')
        for k, v in parameters.items():
            print(f'{k:<25} {v}')
        print()

    # Fine-tuning main function call
    if args.cand_detection:
        cdet_finetune_function.main(subjects, finetune_params, perform_augmentation=args.data_augmentation, save_checkpoint=True, save_weights=save_weights, save_case=args.cp_save_type, verbose=args.verbose, model_directory=model_directory, checkpoint_directory=output_directory)

    if args.verbose:
        print('Finetuned CDet.')

    if args.cand_discrimination:
        cdisc_finetune_function.main(subjects, finetune_params, aug=args.data_augmentation, save_cp=True, save_wei=save_weights, save_case=args.cp_save_type, verbose=args.verbose, model_dir=model_directory, dir_cp=output_directory)

    if args.verbose:
        print('Finetuned CDisc.')
        print('Finetuning complete!')


####################################################################################
# Define the loo_validate (leave-one-out validation) sub-command for microbleednet #
####################################################################################

def cross_validate(args):
    """
    :param args: Input arguments from argparse
    """
    # Usual sanity check for checking if filepaths and files exist.
    preprocessed_directory = args.inp_dir
    # label_directory = args.label_dir
    output_directory = args.output_dir

    if not os.path.isdir(preprocessed_directory):
        raise ValueError(f'{preprocessed_directory} does not appear to be a valid input directory')
    
    input_directory = os.path.join(preprocessed_directory, 'images')
    label_directory = os.path.join(preprocessed_directory, 'labels')
    frst_directory = os.path.join(preprocessed_directory, 'frsts')

    input_paths = glob(os.path.join(input_directory, '*_preproc.nii*'))

    # Check if input directory actually contains files
    if len(input_paths) == 0:
        raise ValueError(f'{input_directory} does not contain any preprocessed input images / filenames NOT in required format')
    
    # Check if label directory is valid
    if os.path.isdir(label_directory) is False:
        raise ValueError(f'{label_directory} does not appear to be a valid directory, please preprocess images')
    
    # Check if FRST directory is valid
    if os.path.isdir(frst_directory) is False:
        raise ValueError(f'{frst_directory} does not appear to be a valid directory, please preprocess images')

    # Check if output directory is valid
    if os.path.isdir(output_directory) is False:
        raise ValueError(f'{output_directory} does not appear to be a valid directory')

    # if os.path.isdir(model_dir) is False:
    #     raise ValueError(model_dir + ' does not appear to be a valid directory')

    # Create a list of dictionaries containing required filepaths for the input subjects
    subjects = []
    for input_path in input_paths:

        basepath = input_path.split('_preproc.nii')[0]
        basename = basepath.split(os.sep)[-1]

        # Checks if the manual label exists for the current file
        label_path = os.path.join(label_directory, basename + '_mask.nii.gz')
        if not os.path.isfile(label_path):
            raise ValueError(f'Manual lesion mask does not exist for {basename}')
        
        # Checks if the FRST exists for the current file
        frst_path = os.path.join(frst_directory, basename + '_frst.nii.gz')
        if not os.path.isfile(frst_path):
            raise ValueError(f'FRST does not exist for {basename}')

        subject = {
            'basename': basename,
            'input_path': input_path,
            'label_path': label_path,
            'frst_path': frst_path,
        }

        subjects.append(subject)

    if isinstance(args.init_learng_rate, float) is False:
        raise ValueError('Initial learning rate must be a float value')
    elif args.init_learng_rate > 1:
        raise ValueError('Initial learning rate must be between 0 and 1')

    if args.optimizer not in ['adam', 'sgd']:
        raise ValueError('Invalid option for Optimizer: Valid options: adam, sgd')

    if isinstance(args.lr_sch_gamma, float) is False:
        raise ValueError('Learning rate reduction factor must be a float value')
    elif args.lr_sch_gamma > 1:
        raise ValueError('Learning rate reduction factor must be between 0 and 1')

    if isinstance(args.train_prop, float) is False:
        raise ValueError('Training data proportion must be a float value')
    elif args.train_prop > 1:
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
    if args.cp_save_type not in ['best', 'last', 'everyN']:
        raise ValueError('Invalid option for checkpoint save type: Valid options: best, last, everyN')
    if args.cp_save_type == 'everyN':
        if args.cp_everyn_N is None:
            raise ValueError('-cp_n must be provided for loading CP when using -cp_type is "everyN"!')

    # This arg does not exist in the argparser
    # if args.num_classes < 1:
    #     raise ValueError('Number of classes to consider in target segmentations must be an int and > 1')
    
    if args.cv_fold < 1:
        raise ValueError('Number of folds cannot be 0 or negative')

    if args.resume_from_fold < 1:
        raise ValueError('Fold to resume cannot be 0 or negative')

    if len(subjects) < args.cv_fold:
        raise ValueError('Number of folds is greater than number of subjects!')

    if args.resume_from_fold > args.cv_fold:
        raise ValueError('The fold to resume CV cannot be higher than the total number of folds specified!')
    
    # Inverts the value stored in args.save_full_model and converts it to bool (this is from original code)
    save_weights = args.save_full_model != 'True'

    # Create the loo_validate parameters dictionary
    crossvalidation_params = {
        'Learning_rate': args.init_learng_rate,
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
        'EveryN': args.cp_everyn_N,
        'SaveResume': args.save_resume_training
    }
    
    if args.verbose:
        parameters = vars(args)
        print('Input parameters are:')
        for k, v in parameters.items():
            print(f'{k:<25} {v}')
        print()

    model_directory = os.path.expandvars('$FSLDIR/data/microbleednet/models')
    if not os.path.exists(model_directory):
        model_directory = os.environ.get('MICROBLEEDNET_PRETRAINED_MODEL_PATH', None)
        if model_directory is None:
            raise RuntimeError('Cannot find teacher model; export MICROBLEEDNET_PRETRAINED_MODEL_PATH=/path/to/your/model')
        
    # Cross-validation main function call
    crossvalidate_function.main(subjects, crossvalidation_params, model_directory, args.data_augmentation, args.intermediate, args.save_checkpoint, save_weights, args.cp_save_type, args.verbose, output_directory, output_directory)

