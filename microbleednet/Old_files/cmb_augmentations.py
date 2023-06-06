# Nicola Dinsdale 2018
# Augment the harp dataset
########################################################################################################################
# Dependencies
import random
from scipy.ndimage.interpolation import shift
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import rotate, zoom
from skimage.util import random_noise
import numpy as np
########################################################################################################################


def translate_it(image, label):
    offsetx = random.randint(-5, 5)
    offsety = random.randint(-5, 5)
    is_seg = False
    order = 0 if is_seg == True else 5
    translated_im = shift(image, (offsetx, offsety, 0), order=order, mode='nearest')
    translated_label = shift(label,(offsetx, offsety, 0), order=order, mode='nearest')
    return translated_im, translated_label


def translate_it_ch2(image, frst, label):
    offsetx = random.randint(-5, 5)
    offsety = random.randint(-5, 5)
    is_seg = False
    order = 0 if is_seg == True else 5
    translated_im = shift(image, (offsetx, offsety, 0), order=order, mode='nearest')
    translated_frst = shift(frst, (offsetx, offsety, 0), order=order, mode='nearest')
    translated_label = shift(label,(offsetx, offsety, 0), order=order, mode='nearest')
    return translated_im, translated_frst, translated_label


def scale_it(image, label):
    factor = random.uniform(0.8, 1.5)
    is_seg = False

    order = 0 if is_seg == True else 3

    height, width, depth = image.shape
    zheight = int(np.round(factor * height))
    zwidth = int(np.round(factor * width))
    zdepth = depth

    if factor < 1.0:
        newimg = np.zeros_like(image)
        newlab = np.zeros_like(label)
        row = (height - zheight) // 2
        col = (width - zwidth) // 2
        layer = (depth - zdepth) // 2
        newimg[row:row+height, col:col+zwidth, layer:layer+zdepth] = zoom(image, (float(factor), float(factor), 1.0),
                                                                          order=order, mode='nearest')[0:zheight, 0:zwidth, 0:zdepth]
        newlab[row:row+height, col:col+zwidth, layer:layer+zdepth] = zoom(label, (float(factor), float(factor), 1.0),
                                                                          order=order, mode='nearest')[0:zheight, 0:zwidth, 0:zdepth]
        return newimg, newlab

    elif factor > 1.0:
        row = (zheight - height) // 2
        col = (zwidth - width) // 2
        layer = (zdepth - depth) // 2

        newimg = zoom(image[row:row+zheight, col:col+zwidth, layer:layer+zdepth], (float(factor), float(factor), 1.0),
                      order=order, mode='nearest')
        newlab = zoom(label[row:row+zheight, col:col+zwidth, layer:layer+zdepth], (float(factor), float(factor), 1.0),
                      order=order, mode='nearest')

        extrah = (newimg.shape[0] - height) // 2
        extraw = (newimg.shape[1] - width) // 2
        extrad = (newimg.shape[2] - depth) // 2
        newimg = newimg[extrah:extrah+height, extraw:extraw+width, extrad:extrad+depth]

        extrah = (newlab.shape[0] - height) // 2
        extraw = (newlab.shape[1] - width) // 2
        extrad = (newlab.shape[2] - depth) // 2
        newlab = newlab[extrah:extrah+height, extraw:extraw+width, extrad:extrad+depth]
        return newimg, newlab

    else:
        return image, label


def rotate_it(image, label):
    theta = random.uniform(-10, 10)
    is_seg = False
    order = 0 if is_seg == True else 5
    new_img = rotate(image, float(theta), axes = (1,2), reshape=False, order=order, mode='nearest')
    new_lab = rotate(label, float(theta), axes = (1,2), reshape=False, order=order, mode='nearest')
    new_lab = (new_lab > 0.6).astype(float)
    return new_img, new_lab


def blur_it(image, label):
    sigma = 0.3
    new_img = gaussian_filter(image, sigma)
    new_lab = gaussian_filter(label, sigma)
    return new_img, new_lab


def flip_it(image, label):
    new_img = image[:, :, ::-1]
    new_lab = label[:, :, ::-1]
    return new_img, new_lab


def flip_it_ch2(image, frst, label):
    new_img = image[:, :, ::-1]
    new_frst = frst[:, :, ::-1]
    new_lab = label[:, :, ::-1]
    return new_img, new_frst, new_lab


def add_noise_to_it_ch2(image, frst, label):
    new_img = random_noise(image)
    new_lab = label
    return new_img, frst, new_lab


def augment(image_to_transform, label):
    # Image applies a random number of the possible transformations to the input image. Returns the transformed image
    # If label is none also applies to the image --> important for segmentation or similar
    """
    :param image_to_transform: input image as array
    :param label: optional: label for input image to also transform
    :return: transformed image, if label also returns transformed label
    """
    if len(image_to_transform.shape) == 3:
        # Add to the available transformations any functions from 3d you want to be applied
        available_transformations = {'flip': flip_it, 'translate': translate_it}
        # Decide how many of these transformations to apply
        num_transformations_to_apply = random.randint(1, len(available_transformations))
        num_transformations = 0
        transformed_image = None
        transformed_label = None

        while num_transformations <= num_transformations_to_apply:
            # choose which transformations to apply at random
            key = random.choice(list(available_transformations))
            transformed_image, transformed_label = available_transformations[key](image_to_transform, label)
            num_transformations += 1
        return transformed_image, transformed_label
    else:
        raise Exception('Invalid dimensions for image augmentation - currently only supported in 3d')


def augment2(image_to_transform, frst, label):
    # Image applies a random number of the possible transformations to the input image. Returns the transformed image
    # If label is none also applies to the image --> important for segmentation or similar
    """
    :param image_to_transform: input image as array
    :param label: optional: label for input image to also transform
    :return: transformed image, if label also returns transformed label
    """
    if len(image_to_transform.shape) == 3:
        # Add to the available transformations any functions from 3d you want to be applied
        available_transformations = {'flip': flip_it_ch2, 'translate': translate_it_ch2,
                                     'add_noise': add_noise_to_it_ch2}
        # Decide how many of these transformations to apply
        num_transformations_to_apply = random.randint(1, len(available_transformations))
        num_transformations = 0
        transformed_image = None
        transformed_frst = None
        transformed_label = None

        while num_transformations <= num_transformations_to_apply:
            # choose which transformations to apply at random
            key = random.choice(list(available_transformations))
            transformed_image, transformed_frst, transformed_label = \
                available_transformations[key](image_to_transform, frst, label)
            num_transformations += 1
        return transformed_image, transformed_frst, transformed_label
    else:
        raise Exception('Invalid dimensions for image augmentation - currently only supported in 3d')
