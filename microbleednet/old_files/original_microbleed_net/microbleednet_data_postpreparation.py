#!/usr/bin/env python
#   Copyright (C) 2020 University of Oxford
#   SHBASECOPYRIGHT

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage.transform import resize
from skimage.measure import regionprops, label
from microbleednet.old_files.original_microbleed_net import (microbleednet_augmentations, microbleednet_data_preprocessing)

#=========================================================================================
# Microbleednet data post-preparation function
# Vaanathi Sundaresan
# 10-01-2023
#=========================================================================================


def preprocess_data(data):
    return (2 * (data / np.amax(np.reshape(data, [-1, 1])))) - 1


def patch_weighing_function(patch):
    temp = np.ones(patch.shape)
    temp[patch.shape[0] // 2 - patch.shape[0] // 3:patch.shape[0] // 2 + patch.shape[0] // 3,
    patch.shape[1] // 2 - patch.shape[1] // 3:patch.shape[1] // 2 + patch.shape[1] // 3,
    patch.shape[2] // 2 - patch.shape[2] // 3:patch.shape[2] // 2 + patch.shape[2] // 3] = 0
    temp_dist = ndimage.morphology.distance_transform_edt(temp)
    temp_dist = np.amax(temp_dist) - temp_dist
    temp_dist = temp_dist / np.amax(temp_dist)
    return temp_dist * temp_dist


def putting_patches_into_images(test_data_paths, prob_patches_all, cent_lists, ps=32):
    prob_volumes = []
    st = 0
    if ps > 32:
        psz = ps//2
    else:
        psz = ps
    print(prob_patches_all.shape)
    for im in range(len(test_data_paths)):
        cents = cent_lists[im]
        prob_patches = prob_patches_all[st:st + cents.shape[0], :, :, :, 1]
        print(cents.shape)
        print(prob_patches.shape)
        st = st + cents.shape[0]
        inp_path = test_data_paths['inp_path']
        data = nib.load(inp_path).get_data().astype(float)
        prob_volume = np.zeros(data.shape)
        crop_data, coords = microbleednet_data_preprocessing.tight_crop_data(data)
        crop_prob_volume = np.zeros(crop_data.shape)
        for c in range(0, cents.shape[0]):
            cmb_x_start = np.amax([int(np.round(cents[c, 0])) - ps // 2, 0])
            cmb_x_end = np.amin([int(np.round(cents[c, 0])) + ps // 2, crop_data.shape[0]])
            cmb_y_start = np.amax([int(np.round(cents[c, 1])) - ps // 2, 0])
            cmb_y_end = np.amin([int(np.round(cents[c, 1])) + ps // 2, crop_data.shape[1]])
            cmb_z_start = np.amax([int(np.round(cents[c, 2])) - psz // 2, 0])
            cmb_z_end = np.amin([int(np.round(cents[c, 2])) + psz // 2, crop_data.shape[2]])
            patch = crop_data[cmb_x_start:cmb_x_end, cmb_y_start:cmb_y_end, cmb_z_start:cmb_z_end]
            patchsize = patch.shape
            patch_weight = patch_weighing_function(patch)
            reqd_patch = patch_weight * prob_patches[c, :patchsize[0], :patchsize[1], :patchsize[2]]
            crop_prob_volume[cmb_x_start:cmb_x_end, cmb_y_start:cmb_y_end, cmb_z_start:cmb_z_end] = np.maximum(
                reqd_patch, crop_prob_volume[cmb_x_start:cmb_x_end, cmb_y_start:cmb_y_end, cmb_z_start:cmb_z_end])
            prob_volume[coords[0]:coords[0] + coords[1], coords[2]:coords[2] + coords[3],
            coords[4]:coords[4] + coords[5]] = np.maximum(crop_prob_volume, prob_volume[coords[0]:coords[0] + coords[1],
                                                                            coords[2]:coords[2] + coords[3],
                                                                            coords[4]:coords[4] + coords[5]])
        prob_volumes.append(prob_volume)
    return prob_volumes


def putting_patches_into_images_frst_ukbb_qsm(test_data_paths, prob_patches_all, ps=32):
    prob_volumes = []
    st = 0
    if ps > 32:
        psz = ps//2
    else:
        psz = ps
    print(prob_patches_all.shape)
    for im in range(len(test_data_paths)):
        inp_path = test_data_paths['inp_path']
        data = nib.load(inp_path).get_data().astype(float)

        #data = resize(data, [data.shape[0] // 2, data.shape[1] // 2, data.shape[2] // 2], preserve_range=True)
        prob_volume = np.zeros(data.shape)
        crop_data, coords = microbleednet_data_preprocessing.tight_crop_data(data)
        crop_prob_volume = np.zeros(crop_data.shape)
        num_patches_x = np.ceil(crop_data.shape[0] / ps).astype(int)
        num_patches_y = np.ceil(crop_data.shape[1] / ps).astype(int)
        num_patches_z = np.ceil(crop_data.shape[2] / psz).astype(int)
        num_patches = num_patches_x * num_patches_y * num_patches_z
        prob_patches = prob_patches_all[st:st + num_patches, :, :, :, 1]
        c = 0
        for cz in range(num_patches_z):
            for cy in range(num_patches_y):
                for cx in range(num_patches_x):
                    cmb_x_start = cx * ps
                    cmb_x_end = np.amin([(cx + 1) * ps, crop_data.shape[0]])
                    if cx == num_patches_x - 1:
                        cmb_x_start = crop_data.shape[0] - ps
                    cmb_y_start = cy * ps
                    cmb_y_end = np.amin([(cy + 1) * ps, crop_data.shape[1]])
                    if cy == num_patches_y - 1:
                        cmb_y_start = crop_data.shape[1] - ps
                    cmb_z_start = cz * psz
                    cmb_z_end = np.amin([(cz + 1) * psz, crop_data.shape[2]])
                    if cz == num_patches_z - 1:
                        cmb_z_start = np.amax([crop_data.shape[2] - psz, 0])
                    patch = crop_data[cmb_x_start:cmb_x_end, cmb_y_start:cmb_y_end, cmb_z_start:cmb_z_end]
                    patchsize = patch.shape
                    reqd_patch = prob_patches[c, :patchsize[0], :patchsize[1], :patchsize[2]]
                    crop_prob_volume[cmb_x_start:cmb_x_end, cmb_y_start:cmb_y_end, cmb_z_start:cmb_z_end] = np.maximum(
                        reqd_patch, crop_prob_volume[cmb_x_start:cmb_x_end, cmb_y_start:cmb_y_end, cmb_z_start:cmb_z_end])
                    prob_volume[coords[0]:coords[0] + coords[1], coords[2]:coords[2] + coords[3], coords[4]:coords[4] + coords[5]] = \
                    np.maximum(crop_prob_volume, prob_volume[coords[0]:coords[0] + coords[1], coords[2]:coords[2] + coords[3],
                                                             coords[4]:coords[4] + coords[5]])
                    c += 1
        prob_volumes.append(prob_volume)
    return prob_volumes


def putting_patches_into_images_frst(test_data_paths, prob_patches_all, ps=32):
    prob_volumes = []
    st = 0
    if ps > 32:
        psz = ps//2
    else:
        psz = ps
    print(prob_patches_all.shape)
    for im in range(len(test_data_paths)):
        inp_path = test_data_paths['inp_path']
        data = nib.load(inp_path).get_data().astype(float)
        data = resize(data, [data.shape[0] // 2, data.shape[1] // 2, data.shape[2] // 2], preserve_range=True)
        prob_volume = np.zeros(data.shape)
        crop_data, coords = microbleednet_data_preprocessing.tight_crop_data(data)
        crop_prob_volume = np.zeros(crop_data.shape)
        num_patches_x = np.ceil(crop_data.shape[0] / ps).astype(int)
        num_patches_y = np.ceil(crop_data.shape[1] / ps).astype(int)
        num_patches_z = np.ceil(crop_data.shape[2] / psz).astype(int)
        num_patches = num_patches_x * num_patches_y * num_patches_z
        prob_patches = prob_patches_all[st:st + num_patches, :, :, :, 1]
        c = 0
        for cz in range(num_patches_z):
            for cy in range(num_patches_y):
                for cx in range(num_patches_x):
                    cmb_x_start = cx * ps
                    cmb_x_end = np.amin([(cx + 1) * ps, crop_data.shape[0]])
                    if cx == num_patches_x - 1:
                        cmb_x_start = crop_data.shape[0] - ps
                    cmb_y_start = cy * ps
                    cmb_y_end = np.amin([(cy + 1) * ps, crop_data.shape[1]])
                    if cy == num_patches_y - 1:
                        cmb_y_start = crop_data.shape[1] - ps
                    cmb_z_start = cz * psz
                    cmb_z_end = np.amin([(cz + 1) * psz, crop_data.shape[2]])
                    if cz == num_patches_z - 1:
                        cmb_z_start = np.amax([crop_data.shape[2] - psz, 0])
                    patch = crop_data[cmb_x_start:cmb_x_end, cmb_y_start:cmb_y_end, cmb_z_start:cmb_z_end]
                    patchsize = patch.shape
                    reqd_patch = prob_patches[c, :patchsize[0], :patchsize[1], :patchsize[2]]
                    crop_prob_volume[cmb_x_start:cmb_x_end, cmb_y_start:cmb_y_end, cmb_z_start:cmb_z_end] = np.maximum(
                        reqd_patch, crop_prob_volume[cmb_x_start:cmb_x_end, cmb_y_start:cmb_y_end, cmb_z_start:cmb_z_end])
                    prob_volume[coords[0]:coords[0] + coords[1], coords[2]:coords[2] + coords[3], coords[4]:coords[4] + coords[5]] = \
                    np.maximum(crop_prob_volume, prob_volume[coords[0]:coords[0] + coords[1], coords[2]:coords[2] + coords[3],
                                                             coords[4]:coords[4] + coords[5]])
                    c += 1
        prob_volumes.append(prob_volume)
    return prob_volumes


def putting_patches_into_images_frst_ukbb(test_data_paths, prob_patches_all, ps=32):
    prob_volumes = []
    st = 0
    if ps > 32:
        psz = ps//2
    else:
        psz = ps
    print(prob_patches_all.shape)
    for im in range(len(test_data_paths)):
        inp_path = test_data_paths[0]['inp_path']
        data = nib.load(inp_path).get_fdata().astype(float)
        #data = resize(data, [data.shape[0] // 2, data.shape[1] // 2, data.shape[2] // 2], preserve_range=True)
        prob_volume = np.zeros(data.shape)
        crop_data, coords = microbleednet_data_preprocessing.tight_crop_data(data)
        crop_prob_volume = np.zeros(crop_data.shape)
        num_patches_x = np.ceil(crop_data.shape[0] / ps).astype(int)
        num_patches_y = np.ceil(crop_data.shape[1] / ps).astype(int)
        num_patches_z = np.ceil(crop_data.shape[2] / psz).astype(int)
        num_patches = num_patches_x * num_patches_y * num_patches_z
        print(prob_patches_all.shape)
        prob_patches = prob_patches_all[st:st + num_patches, :, :, :, 1]
        c = 0
        for cz in range(num_patches_z):
            for cy in range(num_patches_y):
                for cx in range(num_patches_x):
                    cmb_x_start = cx * ps
                    cmb_x_end = np.amin([(cx + 1) * ps, crop_data.shape[0]])
                    if cx == num_patches_x - 1:
                        cmb_x_start = crop_data.shape[0] - ps
                    cmb_y_start = cy * ps
                    cmb_y_end = np.amin([(cy + 1) * ps, crop_data.shape[1]])
                    if cy == num_patches_y - 1:
                        cmb_y_start = crop_data.shape[1] - ps
                    cmb_z_start = cz * psz
                    cmb_z_end = np.amin([(cz + 1) * psz, crop_data.shape[2]])
                    if cz == num_patches_z - 1:
                        cmb_z_start = np.amax([crop_data.shape[2] - psz, 0])
                    patch = crop_data[cmb_x_start:cmb_x_end, cmb_y_start:cmb_y_end, cmb_z_start:cmb_z_end]
                    patchsize = patch.shape
                    reqd_patch = prob_patches[c, :patchsize[0], :patchsize[1], :patchsize[2]]
                    crop_prob_volume[cmb_x_start:cmb_x_end, cmb_y_start:cmb_y_end, cmb_z_start:cmb_z_end] = np.maximum(
                        reqd_patch, crop_prob_volume[cmb_x_start:cmb_x_end, cmb_y_start:cmb_y_end, cmb_z_start:cmb_z_end])
                    prob_volume[coords[0]:coords[0] + coords[1], coords[2]:coords[2] + coords[3], coords[4]:coords[4] + coords[5]] = \
                    np.maximum(crop_prob_volume, prob_volume[coords[0]:coords[0] + coords[1], coords[2]:coords[2] + coords[3],
                                                             coords[4]:coords[4] + coords[5]])
                    c += 1
        prob_volumes.append(prob_volume)
    return prob_volumes


def putting_outputs_into_images_frst(test_data_path, probmap3d):
    print(probmap3d.shape)
    inp_path = test_data_path[0]['inp_path']
    data = nib.load(inp_path).get_fdata()
    prob_volume = np.zeros(data.shape)
    crop_data, coords = microbleednet_data_preprocessing.tight_crop_data(data)
    print(prob_volume.shape)
    print(probmap3d.shape)
    prob_volume[coords[0]:coords[0] + coords[1], coords[2]:coords[2] + coords[3], coords[4]:coords[4] + coords[5]] = probmap3d[0, 1, :, :, :]
    return prob_volume


def putting_patches_into_images_frst_ukbb_testing(test_data_paths, bin_map, prob_patches_all, ps=32):
    prob_volumes = []
    prob_patches_all - np.reshape(prob_patches_all, [-1, 1])
    st = 0
    if ps > 32:
        psz = ps//2
    else:
        psz = ps
    print(prob_patches_all.shape)
    for im in range(len(test_data_paths)):
        inp_path = test_data_paths[0]['inp_path']
        data = nib.load(inp_path).get_fdata().astype(float)
        #data = resize(data, [data.shape[0] // 2, data.shape[1] // 2, data.shape[2] // 2], preserve_range=True)
        distprops = regionprops(label(bin_map > 0))
        cents = np.zeros([len(distprops), 3])
        prob_volume = np.zeros(data.shape)

        print(prob_patches_all.shape, prob_patches_all[0].shape)
        prob_patches = prob_patches_all[st:st + len(distprops)]
        for c in range(len(distprops)):
            cents[c, :] = distprops[c].centroid
            sx = np.amax([int(np.round(cents[c, 0])) - ps // 2, 0])
            ex = np.amin([int(np.round(cents[c, 0])) + ps // 2, bin_map.shape[0]])
            sy = np.amax([int(np.round(cents[c, 1])) - ps // 2, 0])
            ey = np.amin([int(np.round(cents[c, 1])) + ps // 2, bin_map.shape[1]])
            sz = np.amax([int(np.round(cents[c, 2])) - psz // 2, 0])
            ez = np.amin([int(np.round(cents[c, 2])) + psz // 2, bin_map.shape[2]])
            patch = data[sx:ex, sy:ey, sz:ez]
            patchsize = patch.shape
            reqd_patch = prob_patches[c] * np.ones([patchsize[0], patchsize[1], patchsize[2]])
            prob_volume[sx:ex, sy:ey, sz:ez] = np.maximum(reqd_patch, prob_volume[sx:ex, sy:ey, sz:ez])
        prob_volumes.append(prob_volume)
    return prob_volumes


def putting_discpatches_into_images_frst_ukbb_testing(test_data_paths, bin_map, prob_patches_all, ps=32):
    prob_volumes = []

    st = 0
    if ps > 32:
        psz = ps//2
    else:
        psz = ps
    print(prob_patches_all.shape)

    inp_path = test_data_paths[0]['inp_path']
    data = nib.load(inp_path).get_fdata().astype(float)
    distprops = regionprops(label(bin_map > 0))
    cents = np.zeros([len(distprops), 3])
    prob_volume = np.zeros(data.shape)
    prob_patches = prob_patches_all[st:st + len(distprops), :, :, :, 1]
    for c in range(len(distprops)):
        cents[c, :] = distprops[c].centroid
        sx = np.amax([int(np.round(cents[c, 0])) - ps // 2, 0])
        ex = np.amin([int(np.round(cents[c, 0])) + ps // 2, bin_map.shape[0]])
        sy = np.amax([int(np.round(cents[c, 1])) - ps // 2, 0])
        ey = np.amin([int(np.round(cents[c, 1])) + ps // 2, bin_map.shape[1]])
        sz = np.amax([int(np.round(cents[c, 2])) - psz // 2, 0])
        ez = np.amin([int(np.round(cents[c, 2])) + psz // 2, bin_map.shape[2]])
        patch = data[sx:ex, sy:ey, sz:ez]
        patchsize = patch.shape
        reqd_patch = prob_patches[c, :patchsize[0], :patchsize[1], :patchsize[2]]
        prob_volume[sx:ex, sy:ey, sz:ez] = np.maximum(reqd_patch, prob_volume[sx:ex, sy:ey, sz:ez])
    prob_volumes.append(prob_volume)
    return prob_volumes
