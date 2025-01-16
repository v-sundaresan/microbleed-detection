from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
from skimage.util import random_noise
from scipy.ndimage import rotate, zoom
from scipy.ndimage.interpolation import shift
from scipy.ndimage.filters import gaussian_filter

########################################
# Microbleednet augmentations function #
# Vaanathi Sundaresan                  #
# 10-01-2023                           #
########################################

# Original code, using shift is overkill for integer shifts
# def translate(image, label):
#     order = 5
#     offsetx = random.randint(-5, 5)
#     offsety = random.randint(-5, 5)
#     translated_image = shift(image, (offsetx, offsety, 0), order=order, mode='nearest')
#     translated_label = shift(label, (offsetx, offsety, 0), order=order, mode='nearest')
#     return translated_image, translated_label

# def translate_pair(image, frst, label):
#     order = 5
#     offsetx = random.randint(-5, 5)
#     offsety = random.randint(-5, 5)
#     translated_frst = shift(frst, (offsetx, offsety, 0), order=order, mode='nearest')
#     translated_image = shift(image, (offsetx, offsety, 0), order=order, mode='nearest')
#     translated_label = shift(label,(offsetx, offsety, 0), order=order, mode='nearest')
#     return translated_image, translated_frst, translated_label

def translate_array(array, offsetx:int, offsety:int):

    # This method onlyworks for integer offsets

    shifted_array = np.roll(array, shift=offsetx, axis=0)
    shifted_array = np.roll(shifted_array, shift=offsety, axis=1) 

    # Zero out the regions that rolled around
    if offsety > 0:
        shifted_array[:offsety, :, :] = 0  # Top rows zeroed
    elif offsety < 0:
        shifted_array[offsety:, :, :] = 0  # Bottom rows zeroed

    if offsetx > 0:
        shifted_array[:, :offsetx, :] = 0  # Left columns zeroed
    elif offsetx < 0:
        shifted_array[:, offsetx:, :] = 0  # Right columns zeroed

    return shifted_array

def translate(image, label):

    offsetx = random.randint(-5, 5)
    offsety = random.randint(-5, 5)

    translated_image = translate_array(image, offsetx, offsety)
    translated_label = translate_array(label, offsetx, offsety)

    return translated_image, translated_label

def translate_pair(image, frst, label):

    offsetx = random.randint(-5, 5)
    offsety = random.randint(-5, 5)

    translated_image = translate_array(image, offsetx, offsety)
    translated_frst = translate_array(frst, offsetx, offsety)
    translated_label = translate_array(label, offsetx, offsety)

    return translated_image, translated_frst, translated_label

def scale(image, label):
    
    order = 3
    factor = random.uniform(0.8, 1.5)

    height, width, depth = image.shape
    zheight = int(np.round(factor * height))
    zwidth = int(np.round(factor * width))
    zdepth = depth

    if factor < 1.0:

        col = (width - zwidth) // 2
        row = (height - zheight) // 2
        layer = (depth - zdepth) // 2

        scaled_image = np.zeros_like(image)
        scaled_label = np.zeros_like(label)

        scaled_image[row:row+height, col:col+zwidth, layer:layer+zdepth] = zoom(image, (float(factor), float(factor), 1.0), order=order, mode='nearest')[0:zheight, 0:zwidth, 0:zdepth]
        scaled_label[row:row+height, col:col+zwidth, layer:layer+zdepth] = zoom(label, (float(factor), float(factor), 1.0), order=order, mode='nearest')[0:zheight, 0:zwidth, 0:zdepth]

        return scaled_image, scaled_label

    elif factor > 1.0:

        row = (zheight - height) // 2
        col = (zwidth - width) // 2
        layer = (zdepth - depth) // 2

        scaled_image = zoom(image[row:row+zheight, col:col+zwidth, layer:layer+zdepth], (float(factor), float(factor), 1.0), order=order, mode='nearest')
        scaled_label = zoom(label[row:row+zheight, col:col+zwidth, layer:layer+zdepth], (float(factor), float(factor), 1.0), order=order, mode='nearest')

        extrah = (scaled_image.shape[0] - height) // 2
        extraw = (scaled_image.shape[1] - width) // 2
        extrad = (scaled_image.shape[2] - depth) // 2
        scaled_image = scaled_image[extrah:extrah+height, extraw:extraw+width, extrad:extrad+depth]

        extrah = (scaled_label.shape[0] - height) // 2
        extraw = (scaled_label.shape[1] - width) // 2
        extrad = (scaled_label.shape[2] - depth) // 2
        scaled_label = scaled_label[extrah:extrah+height, extraw:extraw+width, extrad:extrad+depth]
        
        return scaled_image, scaled_label

    else:
        return image, label

def rotate(image, label):

    order = 5
    theta = random.uniform(-10, 10)

    rotated_image = rotate(image, float(theta), axes = (1,2), reshape=False, order=order, mode='nearest')
    rotated_label = rotate(label, float(theta), axes = (1,2), reshape=False, order=order, mode='nearest')
    rotated_label = (label > 0.6).astype(float)

    return rotated_image, rotated_label

def blur(image, label):

    sigma = 0.3

    blurred_image = gaussian_filter(image, sigma)
    blurred_label = gaussian_filter(label, sigma)

    return blurred_image, blurred_label

def flip(image, label):

    flipped_image = np.flip(image, axis=2)
    flipped_label = np.flip(label, axis=2)

    return flipped_image, flipped_label

def flip_pair(image, frst, label):

    flipped_frst = np.flip(frst, axis=2)
    flipped_image = np.flip(image, axis=2)
    flipped_label = np.flip(label, axis=2)

    return flipped_image, flipped_frst, flipped_label

def add_noise_pair(image, frst, label):

    noisy_image = random_noise(image)

    return noisy_image, frst, label

def augment(image, label):

    """
    Image applies a random number of the possible transformations to the input image. Returns the transformed image.
    If label is none also applies to the image --> important for segmentation or similar.
    :param image: input image as array
    :param label: optional: label for input image to also transform
    :return: transformed image, if label also returns transformed label
    """

    if len(image.shape) == 3:
        # Add to the available transformations any functions from 3d you want to be applied
        available_transformations = {'flip': flip, 'translate': translate}
        # Decide how many of these transformations to apply
        num_transformations_to_apply = random.randint(1, len(available_transformations))
        transformations = random.sample(list(available_transformations.items()), num_transformations_to_apply)
        
        transformed_image = image
        transformed_label = label

        # for _ in range(num_transformations_to_apply + 1):
        #     # choose which transformations to apply at random
        #     key = random.choice(list(available_transformations))
        #     transformed_image, transformed_label = available_transformations[key](image, label)

        for name, func in transformations:
            transformed_image, transformed_label = func(transformed_image, transformed_label)

        return transformed_image, transformed_label
    
    else:
        raise Exception('Invalid dimensions for image augmentation - currently only supported in 3d')


def augment_pair(image, frst, label):
    
    """
    Image applies a random number of the possible transformations to the input image. Returns the transformed image.
    If label is none also applies to the image --> important for segmentation or similar.
    :param image_to_transform: input image as array
    :param label: optional: label for input image to also transform
    :return: transformed image, if label also returns transformed label
    """
    
    if len(image.shape) == 3:
        # Add to the available transformations any functions from 3d you want to be applied
        available_transformations = {'flip': flip_pair, 'translate': translate_pair, 'add_noise': add_noise_pair}
        # Decide how many of these transformations to apply
        num_transformations_to_apply = random.randint(1, len(available_transformations))
        transformations = random.sample(list(available_transformations.items()), num_transformations_to_apply)

        transformed_frst = frst
        transformed_image = image
        transformed_label = label

        for name, func in transformations:
            transformed_image, transformed_frst, transformed_label = func(transformed_image, transformed_frst, transformed_label)

        return transformed_image, transformed_frst, transformed_label
    
    else:
        raise Exception('Invalid dimensions for image augmentation - currently only supported in 3d')
