from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from microbleednetnet.true_net import (microbleednet_augmentations, microbleednet_data_preprocessing)
from skimage.transform import resize
import nibabel as nib
import numpy as np
from skimage.measure import regionprops, label
import nibabel as nib
from skimage.morphology import dilation, erosion, ball
from skimage import morphology
from skimage import measure
from utils import *
from sklearn.cluster import KMeans
from skimage.restoration import inpaint
from scipy.ndimage import filters
import random
from skimage.filters import frangi
import cmb_augmentations
from skimage.transform import resize
import preprocessing
from scipy.ndimage.morphology import binary_dilation, binary_erosion

#=========================================================================================
# Microbleednet data preparation function
# Vaanathi Sundaresan
# 10-01-2023
#=========================================================================================


def select_train_val_names(data_path, val_numbers):
    """
    :param data_path: list of dictionaries containing filepaths
    :param val_numbers: array of indices
    :return: dictionary of input arrays
    """
    val_ids = random.choices(list(np.arange(len(data_path))), k=val_numbers)
    train_ids = np.setdiff1d(np.arange(len(data_path)), val_ids)
    data_path_train = [data_path[ind] for ind in train_ids]
    data_path_val = [data_path[ind] for ind in val_ids]
    return data_path_train, data_path_val, val_ids


def getting_cmb_manual_patches_fw(ref_man, swi, brain, frst=None, ps=32):
    """
    :param ref_man: input ndarray
    :param swi: input susceptibility-weighted imaging ndarray
    :param brain: brain ndarray
    :param frst: (optional) input ndarray
    :param ps: input scalar
    :return: dictionary of ndarrays
    """
    brain_data1 = erosion(brain, ball(2))
    if ps > 32:
        psz = ps // 2
    else:
        psz = ps
    swi = swi * brain_data1
    ref_man = (ref_man > 0).astype(int)
    num_patches_x = np.ceil(swi.shape[0] / ps).astype(int)
    num_patches_y = np.ceil(swi.shape[1] / ps).astype(int)
    num_patches_z = np.ceil(swi.shape[2] / psz).astype(int)
    num_patches = num_patches_x * num_patches_y * num_patches_z
    cmb_patches = np.zeros([num_patches, ps, ps, psz])
    cmb_man_patches = np.zeros([num_patches, ps, ps, psz])
    if frst is not None:
        frst_patches = np.zeros([num_patches, ps, ps, psz])
    c = 0
    subject_cmb_labs = np.zeros([num_patches, 2])
    cent_patches = np.zeros([num_patches, 3])
    for cz in range(num_patches_z):
        for cy in range(num_patches_y):
            for cx in range(num_patches_x):
                cmb_x_start = cx * ps
                cmb_x_end = np.amin([(cx + 1) * ps, swi.shape[0]])
                if cx == num_patches_x - 1:
                    cmb_x_start = swi.shape[0] - ps
                cmb_y_start = cy * ps
                cmb_y_end = np.amin([(cy + 1) * ps, swi.shape[1]])
                if cy == num_patches_y - 1:
                    cmb_y_start = swi.shape[1] - ps
                cmb_z_start = cz * psz
                cmb_z_end = np.amin([(cz + 1) * psz, swi.shape[2]])
                if cz == num_patches_z - 1:
                    cmb_z_start = swi.shape[2] - psz
                # print(cmb_x_start, cmb_x_end, cmb_y_start, cmb_y_end, cmb_z_start, cmb_z_end)
                cent_patches[c, ...] = [(cmb_x_end - cmb_x_start) / 2, (cmb_y_end - cmb_y_start) / 2,
                                        (cmb_z_end - cmb_z_start) / 2]
                patch = swi[cmb_x_start:cmb_x_end, cmb_y_start:cmb_y_end, cmb_z_start:cmb_z_end]
                mpatch = ref_man[cmb_x_start:cmb_x_end, cmb_y_start:cmb_y_end, cmb_z_start:cmb_z_end]
                # print(patch.shape)
                cmb_patches[c, :patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
                cmb_man_patches[c, :patch.shape[0], :patch.shape[1], :patch.shape[2]] = mpatch
                if np.sum(mpatch) > 0:
                    subject_cmb_labs[c, 1] = 1
                else:
                    subject_cmb_labs[c, 0] = 1

                if frst is not None:
                    frst_patches[c, :patch.shape[0], :patch.shape[1], :patch.shape[2]] = frst[cmb_x_start:cmb_x_end,
                                                                                         cmb_y_start:cmb_y_end,
                                                                                         cmb_z_start:cmb_z_end]
                c += 1

    if frst is not None:
        cmb_patches = np.concatenate([cmb_patches[..., np.newaxis], frst_patches[..., np.newaxis]], axis=-1)
    cmb_cmb_pw = filters.gaussian_filter(cmb_man_patches, 1.2) * 10

    return cmb_patches, cmb_man_patches, cmb_cmb_pw, subject_cmb_labs, cent_patches


def getting_cmb_manual_patches(ref_man, swi, brain, ps=32):
    """
    :param ref_man: input ndarray
    :param swi: input susceptibility-weighted imaging ndarray
    :param brain: brain ndarray
    :param ps: input scalar
    :return: dictionary of ndarrays
    """
    brain_data1 = erosion(brain, ball(2))
    if ps > 32:
        psz = ps // 2
    else:
        psz = ps
    swi = swi * brain_data1
    ref_man = (ref_man > 0).astype(int)
    reqd_man = ref_man > 0
    if np.sum(ref_man) > 0:
        label_man, n = label(reqd_man, return_num=True)
        props = regionprops(label_man)
        cents = np.array([prop.centroid for prop in props])
        cmb_patches = np.zeros([n, ps, ps, psz])
        cmb_man_patches = np.zeros([n, ps, ps, psz])
        for c in range(0, cents.shape[0]):
            cmb_x_start = np.amax([int(np.round(cents[c, 0])) - ps // 2, 0])
            cmb_x_end = np.amin([int(np.round(cents[c, 0])) + ps // 2, reqd_man.shape[0]])
            cmb_y_start = np.amax([int(np.round(cents[c, 1])) - ps // 2, 0])
            cmb_y_end = np.amin([int(np.round(cents[c, 1])) + ps // 2, reqd_man.shape[1]])
            cmb_z_start = np.amax([int(np.round(cents[c, 2])) - psz // 2, 0])
            cmb_z_end = np.amin([int(np.round(cents[c, 2])) + psz // 2, reqd_man.shape[2]])
            patch = swi[cmb_x_start:cmb_x_end, cmb_y_start:cmb_y_end, cmb_z_start:cmb_z_end]
            mpatch = ref_man[cmb_x_start:cmb_x_end, cmb_y_start:cmb_y_end, cmb_z_start:cmb_z_end]
            tmp_patch = np.zeros([ps, ps, psz])
            tmp_patch[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
            cmb_patches[c, :, :, :] = tmp_patch
            tmp_manpatch = np.zeros([ps, ps, psz])
            tmp_manpatch[:mpatch.shape[0], :mpatch.shape[1], :mpatch.shape[2]] = mpatch
            cmb_man_patches[c, :, :, :] = tmp_manpatch
        ref_img = ((ref_man == 0).astype(int)) * ((swi > -0.8).astype(int))
        pos = np.array(np.where(ref_img))
        max_samples = n * 2
        if n == 0:
            max_samples = 5
    else:
        ref_img = ((ref_man == 0).astype(int)) * ((swi > -0.8).astype(int))
        pos = np.array(np.where(ref_img))
        max_samples = 10
    passed_non_cmb_cent_ids = np.random.choice(pos.shape[1], max_samples, replace=False)
    passed_noncmb_cents = pos[:, passed_non_cmb_cent_ids]
    non_cmb_patches = np.zeros([passed_noncmb_cents.shape[1], ps, ps, psz])
    del pos
    for t in range(0, passed_noncmb_cents.shape[1]):
        noncmb_x_start = np.amax([passed_noncmb_cents[0, t] - ps // 2, 0])
        noncmb_x_end = np.amin([passed_noncmb_cents[0, t] + ps // 2, ref_man.shape[0]])
        noncmb_y_start = np.amax([passed_noncmb_cents[1, t] - ps // 2, 0])
        noncmb_y_end = np.amin([passed_noncmb_cents[1, t] + ps // 2, ref_man.shape[1]])
        noncmb_z_start = np.amax([passed_noncmb_cents[2, t] - psz // 2, 0])
        noncmb_z_end = np.amin([passed_noncmb_cents[2, t] + psz // 2, ref_man.shape[2]])
        npatch = swi[noncmb_x_start:noncmb_x_end, noncmb_y_start:noncmb_y_end, noncmb_z_start:noncmb_z_end]
        if npatch.shape[0] < ps or npatch.shape[1] < ps or npatch.shape[2] < psz:
            tpatch = np.zeros([ps, ps, psz])
            tpatch[:npatch.shape[0], :npatch.shape[1], :npatch.shape[2]] = npatch
            noncmb_patch = tpatch
        else:
            noncmb_patch = npatch
        non_cmb_patches[t, :, :, :] = noncmb_patch
    non_cmb_man_patches = 0 * non_cmb_patches
    cmb_noncmb_pw = filters.gaussian_filter(non_cmb_man_patches, 1.2) * 10
    passed_noncmb_cents = passed_noncmb_cents.transpose(1, 0)
    non_cmb_labs = np.hstack([np.zeros([non_cmb_patches.shape[0], 1]), np.ones([non_cmb_patches.shape[0], 1])])
    if np.sum(ref_man) > 0:
        cmb_cmb_pw = filters.gaussian_filter(cmb_man_patches, 1.2) * 10
        data_patches = np.concatenate((cmb_patches, non_cmb_patches), axis=0)
        labels_patches = np.concatenate((cmb_man_patches, non_cmb_man_patches), axis=0)
        pw_patches = np.concatenate((cmb_cmb_pw, cmb_noncmb_pw), axis=0)
        cent_patches = np.vstack([cents, passed_noncmb_cents])
        cmb_labs = np.hstack([np.zeros([cmb_patches.shape[0], 1]), np.ones([cmb_patches.shape[0], 1])])
        subjlabs_patches = np.vstack([cmb_labs, non_cmb_labs])
    else:
        data_patches = non_cmb_patches
        labels_patches = non_cmb_man_patches
        pw_patches = cmb_noncmb_pw
        cent_patches = passed_noncmb_cents
        subjlabs_patches = non_cmb_labs
    return data_patches, labels_patches, pw_patches, subjlabs_patches, cent_patches


def getting_cmb_test_patches(ref_man, swi, brain, ps=32, pmap=None):
    """
    :param ref_man: input ndarray
    :param swi: input susceptibility-weighted imaging ndarray
    :param brain: brain ndarray
    :param ps: input scalar
    :param pmap: input priormap ndarray
    :return: dictionary of ndarrays
    """
    brain_data1 = erosion(brain, ball(2))
    ero_data1 = swi * brain_data1
    # ero_data1[ero_data1 == 0] = 0.5
    swi2 = filters.gaussian_filter(ero_data1, 0.4)
    if ps > 32:
        psz = ps // 2
    else:
        psz = ps
    # ca = []
    if pmap is None:
        loc_maxima = morphology.local_maxima(swi2)
        labs, num_labs = morphology.label(loc_maxima, return_num=True)
        props = measure.regionprops(labs)
        cent = np.array([prop.centroid for prop in props])
        ids = []
        for idx in range(cent.shape[0]):
            if swi2[int(cent[idx, 0]), int(cent[idx, 1]), int(cent[idx, 2])] > 0.75:
                ids.append(idx)
        cent = cent[ids, :]
        new_img = np.zeros([swi2.shape[0], swi2.shape[1], swi2.shape[2]])
        for idx in range(cent.shape[0]):
            new_img[int(cent[idx, 0]), int(cent[idx, 1]), int(cent[idx, 2])] = 1
        new_img1 = dilation(new_img, ball(1))
        labs, num_labs = morphology.label(new_img1 > 0, return_num=True)
        props = measure.regionprops(labs)
        cents = np.array([prop.centroid for prop in props])
    else:
        lmax = morphology.h_maxima(pmap * brain, 0.2)
        llab, num_labs = morphology.label(lmax > 0, return_num=True)
        lprops = regionprops(llab)
        cents = np.array([prop.centroid for prop in lprops])
    cmb_patches = np.zeros([num_labs, ps, ps, psz])
    cmb_man_patches = np.zeros([num_labs, ps, ps, psz])
    for c in range(0, cents.shape[0]):
        cmb_x_start = np.amax([int(np.round(cents[c, 0])) - ps // 2, 0])
        cmb_x_end = np.amin([int(np.round(cents[c, 0])) + ps // 2, ref_man.shape[0]])
        cmb_y_start = np.amax([int(np.round(cents[c, 1])) - ps // 2, 0])
        cmb_y_end = np.amin([int(np.round(cents[c, 1])) + ps // 2, ref_man.shape[1]])
        cmb_z_start = np.amax([int(np.round(cents[c, 2])) - psz // 2, 0])
        cmb_z_end = np.amin([int(np.round(cents[c, 2])) + psz // 2, ref_man.shape[2]])
        patch = swi[cmb_x_start:cmb_x_end, cmb_y_start:cmb_y_end, cmb_z_start:cmb_z_end]
        mpatch = ref_man[cmb_x_start:cmb_x_end, cmb_y_start:cmb_y_end, cmb_z_start:cmb_z_end]
        tmp_patch = np.zeros([ps, ps, psz])
        tmp_patch[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
        cmb_patches[c, :, :, :] = tmp_patch
        tmp_manpatch = np.zeros([ps, ps, psz])
        tmp_manpatch[:mpatch.shape[0], :mpatch.shape[1], :mpatch.shape[2]] = mpatch
        cmb_man_patches[c, :, :, :] = tmp_manpatch
    cmb_pw = filters.gaussian_filter(cmb_man_patches, 1.2) * 10
    subjlabs_patches = np.zeros([cmb_pw.shape[0], 2])
    return cmb_patches, cmb_man_patches, cmb_pw, subjlabs_patches, cents


def getting_cmb_test_patches_fw(swi, brain, frst=None, ps=32, pmap=None):
    """
    :param swi: input susceptibility-weighted imaging ndarray
    :param brain: brain ndarray
    :param frst: input ndarray
    :param ps: input scalar
    :param pmap: input priormap ndarray
    :return: dictionary of ndarrays
    """
    brain_data1 = erosion(brain, ball(2))
    if ps > 32:
        psz = ps // 2
    else:
        psz = ps
    swi = swi * brain_data1
    num_patches_x = np.ceil(swi.shape[0] / ps).astype(int)
    num_patches_y = np.ceil(swi.shape[1] / ps).astype(int)
    num_patches_z = np.ceil(swi.shape[2] / psz).astype(int)
    num_patches = num_patches_x * num_patches_y * num_patches_z
    cmb_patches = np.zeros([num_patches, ps, ps, psz])
    cmb_man_patches = np.zeros([num_patches, ps, ps, psz])
    if frst is not None:
        frst_patches = np.zeros([num_patches, ps, ps, psz])
    c = 0
    cent_patches = np.zeros([num_patches, 3])
    for cz in range(num_patches_z):
        for cy in range(num_patches_y):
            for cx in range(num_patches_x):
                cmb_x_start = cx * ps
                cmb_x_end = np.amin([(cx + 1) * ps, swi.shape[0]])
                if cx == num_patches_x - 1:
                    cmb_x_start = swi.shape[0] - ps
                cmb_y_start = cy * ps
                cmb_y_end = np.amin([(cy + 1) * ps, swi.shape[1]])
                if cy == num_patches_y - 1:
                    cmb_y_start = swi.shape[1] - ps
                cmb_z_start = cz * psz
                cmb_z_end = np.amin([(cz + 1) * psz, swi.shape[2]])
                if cz == num_patches_z - 1:
                    cmb_z_start = np.amax([swi.shape[2] - psz, 0])

                cent_patches[c, ...] = [(cmb_x_end - cmb_x_start) / 2, (cmb_y_end - cmb_y_start) / 2,
                                        (cmb_z_end - cmb_z_start) / 2]
                patch = swi[cmb_x_start:cmb_x_end, cmb_y_start:cmb_y_end, cmb_z_start:cmb_z_end]
                cmb_patches[c, :patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
                cmb_man_patches[c, :patch.shape[0], :patch.shape[1], :patch.shape[2]] = np.zeros_like(patch)
                if frst is not None:
                    frst_patches[c, :patch.shape[0], :patch.shape[1], :patch.shape[2]] = frst[cmb_x_start:cmb_x_end,
                                                                                         cmb_y_start:cmb_y_end,
                                                                                         cmb_z_start:cmb_z_end]
                c += 1

    if frst is not None:
        cmb_patches = np.concatenate([cmb_patches[..., np.newaxis], frst_patches[..., np.newaxis]], axis=-1)
    cmb_cmb_pw = filters.gaussian_filter(cmb_man_patches, 1.2) * 10

    return cmb_patches, cmb_man_patches, cmb_cmb_pw, cent_patches


def load_and_prepare_cmb_data_frst(filename, train='train', ps=32, priormap=None):
    """
    :param filename: list of dictionaries containing filepaths
    :param train: string; mode for loading
    :param ps: input scalar
    :param priormap: input prior map ndarray
    :return: dictionary of ndarrays
    """
    patch_data = np.array([])
    patch_labels = np.array([])
    patch_pws = np.array([])
    for im in range(len(filename)):
        data = nib.load(filename[im]).get_data().astype(float)
        data = resize(data, [data.shape[0] // 2, data.shape[1] // 2, data.shape[2] // 2], preserve_range=True)
        try:
            label = nib.load(filename[im][:-22] + 'CMB.nii.gz').get_data()
            label = resize(label, [label.shape[0] // 2, label.shape[1] // 2, label.shape[2] // 2],
                           preserve_range=True)
            label = (label > 32768).astype(int)
        except:
            label = np.zeros_like(data, dtype=int)
        # brain = nib.load(filename[im][:-7] + '_brain_mask.nii.gz').get_data().astype(int)
        # brain = resize(brain, [brain.shape[0] // 2, brain.shape[1] // 2, brain.shape[2] // 2],
        #                preserve_range=True)
        brain = (data > 0).astype(int)
        frst = np.load(filename[im][:-7] + '_frst.npy')
        frst[np.isnan(frst)] = 0
        crop_org_data, coords = microbleednet_data_preprocessing.tight_crop_data(data)
        data1 = 1 - (data / np.amax(np.reshape(data, [-1, 1])))
        data1 = data1 * brain.astype(float)
        data1 = microbleednet_data_preprocessing.preprocess_data(data1)

        crop_data = data1[coords[0]:coords[0] + coords[1], coords[2]:coords[2] + coords[3],
                    coords[4]:coords[4] + coords[5]]
        labels = label[coords[0]:coords[0] + coords[1], coords[2]:coords[2] + coords[3],
                 coords[4]:coords[4] + coords[5]]
        crop_brain = brain[coords[0]:coords[0] + coords[1], coords[2]:coords[2] + coords[3],
                     coords[4]:coords[4] + coords[5]]
        crop_frst = frst[coords[0]:coords[0] + coords[1], coords[2]:coords[2] + coords[3],
                    coords[4]:coords[4] + coords[5]]
        crop_data[crop_org_data == 0] = 0.5
        reqd_man = labels > 0
        if train == 'train':
            if ps < 48:
                patchdata, patchlabels, patchpw, _, _ = getting_cmb_manual_patches_fw(reqd_man,
                                                                                      crop_data,
                                                                                      crop_brain,
                                                                                      frst=crop_frst,
                                                                                      ps=ps)
            else:
                patchdata, patchlabels, patchpw, _, _ = getting_cmb_manual_patches_fw(reqd_man,
                                                                                      crop_data,
                                                                                      crop_brain,
                                                                                      frst=crop_frst,
                                                                                      ps=ps)
            cmb_train = np.copy(patchdata)
            if len(cmb_train.shape) > 4:
                chn2 = 1
            else:
                chn2 = 0

            if ps > 32:
                psz = ps // 2
            else:
                psz = ps
            augmented_img_list = []
            augmented_mseg_list = []
            if chn2:
                augmented_frst_list = []
            for idx in range(0, cmb_train.shape[0]):
                for i in range(0, 4):
                    manmask = patchlabels[idx, :, :, :]
                    if chn2:
                        image = cmb_train[idx, :, :, :, 0]
                        frst = cmb_train[idx, :, :, :, 1]
                        augmented_img, augmented_frst, augmented_manseg = \
                            microbleednet_augmentations.augment2(image, frst, manmask)
                    else:
                        image = cmb_train[idx, :, :, :]
                        augmented_img, augmented_manseg = microbleednet_augmentations.augment(image, manmask)
                    augmented_img_list.append(augmented_img)
                    augmented_mseg_list.append(augmented_manseg)
                    if chn2:
                        augmented_frst_list.append(augmented_frst)

            augmented_mseg = np.array(augmented_mseg_list)
            augmented_mseg = np.reshape(augmented_mseg, [-1, ps, ps, psz])
            if chn2:
                augmented_frst = np.array(augmented_frst_list)
                augmented_frsts = np.reshape(augmented_frst, [-1, ps, ps, psz, 1])
                augmented_img = np.array(augmented_img_list)
                augmented_imgs = np.reshape(augmented_img, [-1, ps, ps, psz, 1])
                augmented_imgs = np.concatenate((augmented_imgs, augmented_frsts), axis=-1)
            else:
                augmented_img = np.array(augmented_img_list)
                augmented_imgs = np.reshape(augmented_img, [-1, ps, ps, psz])

            cmb_train = np.concatenate((cmb_train, augmented_imgs), axis=0)
            cmb_train_labs = np.concatenate((patchlabels, augmented_mseg), axis=0)
            if chn2:
                patchdata = cmb_train
            else:
                cmb_train = np.tile(cmb_train, (1, 1, 1, 1, 1))
                patchdata = cmb_train.transpose(1, 2, 3, 4, 0)
            patchdata[patchdata < 0] = 0
            patchpw = filters.gaussian_filter(cmb_train_labs, 1.2) * 10
            patch_data = np.concatenate((patch_data, patchdata), axis=0) if patch_data.size else patchdata
            patch_labels = np.concatenate((patch_labels, cmb_train_labs),
                                          axis=0) if patch_labels.size else cmb_train_labs
            patch_pws = np.concatenate((patch_pws, patchpw), axis=0) if patch_pws.size else patchpw
        elif train == 'val':
            if ps < 48:
                patchdata, patchlabels, patchpw, _, _ = getting_cmb_manual_patches_fw(reqd_man,
                                                                                      crop_data,
                                                                                      crop_brain,
                                                                                      frst=crop_frst,
                                                                                      ps=ps)
            else:
                patchdata, patchlabels, patchpw, _, _ = getting_cmb_manual_patches_fw(reqd_man,
                                                                                      crop_data,
                                                                                      crop_brain,
                                                                                      frst=crop_frst,
                                                                                      ps=ps)
            if len(patchdata.shape) > 4:
                chn2 = 1
            else:
                chn2 = 0

            if chn2 == 0:
                patchdata = np.tile(patchdata, (1, 1, 1, 1, 1))
                patchdata = patchdata.transpose(1, 2, 3, 4, 0)

            patchdata[patchdata < 0] = 0
            patch_data = np.concatenate((patch_data, patchdata), axis=0) if patch_data.size else patchdata
            patch_labels = np.concatenate((patch_labels, patchlabels), axis=0) if patch_labels.size else patchlabels
            patch_pws = np.concatenate((patch_pws, patchpw), axis=0) if patch_pws.size else patchpw
        else:
            if priormap is not None:
                priormap = priormap[coords[0]:coords[0] + coords[1], coords[2]:coords[2] + coords[3],
                           coords[4]:coords[4] + coords[5]]
            patchdata, patchlabels, patchpw, patchcents = getting_cmb_test_patches_fw(crop_data,
                                                                                      crop_brain, frst=crop_frst,
                                                                                      ps=ps, pmap=priormap)
            if len(patchdata.shape) > 4:
                chn2 = 1
            else:
                chn2 = 0

            if chn2 == 0:
                patchdata = np.tile(patchdata, (1, 1, 1, 1, 1))
                patchdata = patchdata.transpose(1, 2, 3, 4, 0)
            patchdata[patchdata < 0] = 0
            patch_data = np.concatenate((patch_data, patchdata), axis=0) if patch_data.size else patchdata
            patch_labels = np.concatenate((patch_labels, patchlabels), axis=0) if patch_labels.size else patchlabels
            patch_pws = np.concatenate((patch_pws, patchpw), axis=0) if patch_pws.size else patchpw

    return [patch_data, patch_labels, patch_pws]


def load_and_prepare_cmb_data_frst_ukbb(filename, train='train', ps=32, priormap=None):
    """
    :param filename: list of dictionaries containing filepaths
    :param train: string; mode for loading
    :param ps: input scalar
    :param priormap: input prior map ndarray
    :return: dictionary of ndarrays
    """
    patch_data = np.array([])
    patch_labels = np.array([])
    patch_pws = np.array([])
    for im in range(len(filename)):
        data = nib.load('path/to/the/input_SWI_nifti.nii.gz').get_data().astype(float)
        # data = resize(data, [data.shape[0] // 2, data.shape[1] // 2, data.shape[2] // 2], preserve_range=True)
        try:
            label = nib.load('path/to/the/input_inplabels_nifti.nii.gz').get_data()
            # label = resize(label, [label.shape[0] // 2, label.shape[1] // 2, label.shape[2] // 2], preserve_range=True)
            label = (label > 0).astype(int)
        except:
            label = np.zeros_like(data, dtype=int)
        brain = (data > 0).astype(int)
        frst = np.load(filename[im])
        frst[np.isnan(frst)] = 0
        crop_org_data, coords = microbleednet_data_preprocessing.tight_crop_data(data)
        data1 = 1 - (data / np.amax(np.reshape(data, [-1, 1])))
        data1 = data1 * brain.astype(float)
        data1 = microbleednet_data_preprocessing.preprocess_data(data1)

        crop_data = data1[coords[0]:coords[0] + coords[1], coords[2]:coords[2] + coords[3],
                    coords[4]:coords[4] + coords[5]]
        labels = label[coords[0]:coords[0] + coords[1], coords[2]:coords[2] + coords[3],
                 coords[4]:coords[4] + coords[5]]
        crop_brain = brain[coords[0]:coords[0] + coords[1], coords[2]:coords[2] + coords[3],
                     coords[4]:coords[4] + coords[5]]
        crop_frst = frst[coords[0]:coords[0] + coords[1], coords[2]:coords[2] + coords[3],
                    coords[4]:coords[4] + coords[5]]
        crop_data[crop_org_data == 0] = 0.5
        reqd_man = labels > 0
        if train == 'train':
            if ps < 48:
                patchdata, patchlabels, patchpw, _, _ = getting_cmb_manual_patches_fw(reqd_man,
                                                                                      crop_data,
                                                                                      crop_brain,
                                                                                      frst=crop_frst,
                                                                                      ps=ps)
            else:
                patchdata, patchlabels, patchpw, _, _ = getting_cmb_manual_patches_fw(reqd_man,
                                                                                      crop_data,
                                                                                      crop_brain,
                                                                                      frst=crop_frst,
                                                                                      ps=ps)
            cmb_train = np.copy(patchdata)
            if len(cmb_train.shape) > 4:
                chn2 = 1
            else:
                chn2 = 0

            if ps > 32:
                psz = ps // 2
            else:
                psz = ps
            augmented_img_list = []
            augmented_mseg_list = []
            if chn2:
                augmented_frst_list = []
            for idx in range(0, cmb_train.shape[0]):
                for i in range(0, 4):
                    manmask = patchlabels[idx, :, :, :]
                    if chn2:
                        image = cmb_train[idx, :, :, :, 0]
                        frst = cmb_train[idx, :, :, :, 1]
                        augmented_img, augmented_frst, augmented_manseg = \
                            microbleednet_augmentations.augment2(image, frst, manmask)
                    else:
                        image = cmb_train[idx, :, :, :]
                        augmented_img, augmented_manseg = microbleednet_augmentations.augment(image, manmask)
                    augmented_img_list.append(augmented_img)
                    augmented_mseg_list.append(augmented_manseg)
                    if chn2:
                        augmented_frst_list.append(augmented_frst)

            augmented_mseg = np.array(augmented_mseg_list)
            augmented_mseg = np.reshape(augmented_mseg, [-1, ps, ps, psz])
            if chn2:
                augmented_frst = np.array(augmented_frst_list)
                augmented_frsts = np.reshape(augmented_frst, [-1, ps, ps, psz, 1])
                augmented_img = np.array(augmented_img_list)
                augmented_imgs = np.reshape(augmented_img, [-1, ps, ps, psz, 1])
                augmented_imgs = np.concatenate((augmented_imgs, augmented_frsts), axis=-1)
            else:
                augmented_img = np.array(augmented_img_list)
                augmented_imgs = np.reshape(augmented_img, [-1, ps, ps, psz])

            cmb_train = np.concatenate((cmb_train, augmented_imgs), axis=0)
            cmb_train_labs = np.concatenate((patchlabels, augmented_mseg), axis=0)
            if chn2:
                patchdata = cmb_train
            else:
                cmb_train = np.tile(cmb_train, (1, 1, 1, 1, 1))
                patchdata = cmb_train.transpose(1, 2, 3, 4, 0)
            patchdata[patchdata < 0] = 0
            patchpw = filters.gaussian_filter(cmb_train_labs, 1.2) * 10
            patch_data = np.concatenate((patch_data, patchdata), axis=0) if patch_data.size else patchdata
            patch_labels = np.concatenate((patch_labels, cmb_train_labs),
                                          axis=0) if patch_labels.size else cmb_train_labs
            patch_pws = np.concatenate((patch_pws, patchpw), axis=0) if patch_pws.size else patchpw
        elif train == 'val':
            if ps < 48:
                patchdata, patchlabels, patchpw, _, _ = getting_cmb_manual_patches_fw(reqd_man,
                                                                                      crop_data,
                                                                                      crop_brain,
                                                                                      frst=crop_frst,
                                                                                      ps=ps)
            else:
                patchdata, patchlabels, patchpw, _, _ = getting_cmb_manual_patches_fw(reqd_man,
                                                                                      crop_data,
                                                                                      crop_brain,
                                                                                      frst=crop_frst,
                                                                                      ps=ps)
            if len(patchdata.shape) > 4:
                chn2 = 1
            else:
                chn2 = 0

            if chn2 == 0:
                patchdata = np.tile(patchdata, (1, 1, 1, 1, 1))
                patchdata = patchdata.transpose(1, 2, 3, 4, 0)

            patchdata[patchdata < 0] = 0
            patch_data = np.concatenate((patch_data, patchdata), axis=0) if patch_data.size else patchdata
            patch_labels = np.concatenate((patch_labels, patchlabels), axis=0) if patch_labels.size else patchlabels
            patch_pws = np.concatenate((patch_pws, patchpw), axis=0) if patch_pws.size else patchpw
        else:
            if priormap is not None:
                priormap = priormap[coords[0]:coords[0] + coords[1], coords[2]:coords[2] + coords[3],
                           coords[4]:coords[4] + coords[5]]
            patchdata, patchlabels, patchpw, patchcents = getting_cmb_test_patches_fw(crop_data,
                                                                                      crop_brain, frst=crop_frst,
                                                                                      ps=ps, pmap=priormap)
            if len(patchdata.shape) > 4:
                chn2 = 1
            else:
                chn2 = 0

            if chn2 == 0:
                patchdata = np.tile(patchdata, (1, 1, 1, 1, 1))
                patchdata = patchdata.transpose(1, 2, 3, 4, 0)
            patchdata[patchdata < 0] = 0
            patch_data = np.concatenate((patch_data, patchdata), axis=0) if patch_data.size else patchdata
            patch_labels = np.concatenate((patch_labels, patchlabels), axis=0) if patch_labels.size else patchlabels
            patch_pws = np.concatenate((patch_pws, patchpw), axis=0) if patch_pws.size else patchpw

    return [patch_data, patch_labels, patch_pws]


def load_and_prepare_cmb_data(filename, train='train', ps=32, priormap=None):
    """
    :param filename: list of dictionaries containing filepaths
    :param train: string; mode for loading
    :param ps: input scalar
    :param priormap: input prior map ndarray
    :return: dictionary of ndarrays
    """
    patch_data = np.array([])
    patch_labels = np.array([])
    patch_pws = np.array([])
    patch_cmblabs = np.array([])
    patch_cents = []
    for im in range(len(filename)):
        data = nib.load(filename[im]).get_data().astype(float)
        data = resize(data, [data.shape[0] // 2, data.shape[1] // 2, data.shape[2] // 2], preserve_range=True)
        try:
            label = nib.load(filename[im][:-22] + 'CMB.nii.gz').get_data()
            label = resize(label, [label.shape[0] // 2, label.shape[1] // 2, label.shape[2] // 2],
                           preserve_range=True)
            label = (label > 32768).astype(int)
        except:
            label = np.zeros_like(data, dtype=int)
        # brain = nib.load(filename[im][:-7] + '_brain_mask.nii.gz').get_data().astype(int)
        # brain = resize(brain, [brain.shape[0] // 2, brain.shape[1] // 2, brain.shape[2] // 2],
        #                preserve_range=True)
        brain = (data > 0).astype(int)
        # print('data dimensions after loading: ', data.shape)
        frst = get_frst_data(data)
        crop_org_data, coords = microbleednet_data_preprocessing.tight_crop_data(data)
        data1 = 1 - (data / np.amax(np.reshape(data, [-1, 1])))
        data1 = data1 * brain.astype(float)
        data1 = microbleednet_data_preprocessing.preprocess_data(data1)

        crop_data = data1[coords[0]:coords[0] + coords[1], coords[2]:coords[2] + coords[3],
                    coords[4]:coords[4] + coords[5]]
        labels = label[coords[0]:coords[0] + coords[1], coords[2]:coords[2] + coords[3],
                 coords[4]:coords[4] + coords[5]]
        crop_brain = brain[coords[0]:coords[0] + coords[1], coords[2]:coords[2] + coords[3],
                     coords[4]:coords[4] + coords[5]]
        crop_frst = frst[coords[0]:coords[0] + coords[1], coords[2]:coords[2] + coords[3],
                    coords[4]:coords[4] + coords[5]]
        crop_data[crop_org_data == 0] = 0.5
        # print('data dimensions after cropping: ', crop_data.shape)
        reqd_man = labels > 0
        if train == 'train':
            if ps < 48:
                if ps == 32:
                    obj = np.load(filename[im][:-10] + 'patch32.npz')
                else:
                    obj = np.load(filename[im][:-10] + 'patch24.npz')
                patchdata = obj['data']
                patchlabels = obj['lab']
                patchpw = obj['pw']
                patchcmblabs = obj['cmblabs']
                patchcents = obj['cents']
            else:
                patchdata, patchlabels, patchpw, patchcmblabs, patchcents = getting_cmb_manual_patches(reqd_man,
                                                                                                       crop_data,
                                                                                                       crop_brain,
                                                                                                       ps)
            cmb_train = np.copy(patchdata)
            if ps > 32:
                psz = ps // 2
            else:
                psz = ps
            augmented_img_list = []
            augmented_mseg_list = []
            for idx in range(0, cmb_train.shape[0]):
                for i in range(0, 2):
                    image = cmb_train[idx, :, :, :]
                    manmask = patchlabels[idx, :, :, :]
                    augmented_img, augmented_manseg = microbleednet_augmentations.augment(image, manmask)
                    augmented_img_list.append(augmented_img)
                    augmented_mseg_list.append(augmented_manseg)

            augmented_img = np.array(augmented_img_list)
            augmented_mseg = np.array(augmented_mseg_list)
            augmented_imgs = np.reshape(augmented_img, [-1, ps, ps, psz])
            augmented_mseg = np.reshape(augmented_mseg, [-1, ps, ps, psz])

            cmb_train = np.concatenate((cmb_train, augmented_imgs), axis=0)
            cmb_train_labs = np.concatenate((patchlabels, augmented_mseg), axis=0)
            cmb_train = np.tile(cmb_train, (1, 1, 1, 1, 1))
            patchdata = cmb_train.transpose(1, 2, 3, 4, 0)
            patchdata[patchdata < 0] = 0
            patchpw = filters.gaussian_filter(cmb_train_labs, 1.2) * 10
            patch_data = np.concatenate((patch_data, patchdata), axis=0) if patch_data.size else patchdata
            patch_labels = np.concatenate((patch_labels, cmb_train_labs),
                                          axis=0) if patch_labels.size else cmb_train_labs
            patch_pws = np.concatenate((patch_pws, patchpw), axis=0) if patch_pws.size else patchpw
            patch_cmblabs = np.concatenate((patch_cmblabs, patchcmblabs),
                                           axis=0) if patch_cmblabs.size else patchcmblabs
            patch_cents.append(patchcents)
        elif train == 'val':
            if ps < 48:
                if ps == 32:
                    obj = np.load(filename[im][:-10] + 'patch32.npz')
                else:
                    obj = np.load(filename[im][:-10] + 'patch24.npz')
                patchdata = obj['data']
                patchlabels = obj['lab']
                patchpw = obj['pw']
                patchcmblabs = obj['cmblabs']
                patchcents = obj['cents']
            else:
                patchdata, patchlabels, patchpw, patchcmblabs, patchcents = getting_cmb_manual_patches(reqd_man,
                                                                                                       crop_data,
                                                                                                       crop_brain,
                                                                                                       ps)
            patchdata = np.tile(patchdata, (1, 1, 1, 1, 1))
            patchdata = patchdata.transpose(1, 2, 3, 4, 0)
            patchdata[patchdata < 0] = 0
            patch_data = np.concatenate((patch_data, patchdata), axis=0) if patch_data.size else patchdata
            patch_labels = np.concatenate((patch_labels, patchlabels), axis=0) if patch_labels.size else patchlabels
            patch_pws = np.concatenate((patch_pws, patchpw), axis=0) if patch_pws.size else patchpw
            patch_cmblabs = np.concatenate((patch_cmblabs, patchcmblabs),
                                           axis=0) if patch_cmblabs.size else patchcmblabs
            patch_cents.append(patchcents)
        else:
            if priormap is not None:
                priormap = priormap[coords[0]:coords[0] + coords[1], coords[2]:coords[2] + coords[3],
                           coords[4]:coords[4] + coords[5]]
            patchdata, patchlabels, patchpw, patchcmblabs, patchcents = getting_cmb_test_patches(reqd_man,
                                                                                                 crop_data,
                                                                                                 crop_brain,
                                                                                                 ps,
                                                                                                 pmap=priormap)
            patchdata = np.tile(patchdata, (1, 1, 1, 1, 1))
            patchdata = patchdata.transpose(1, 2, 3, 4, 0)
            patchdata[patchdata < 0] = 0
            patch_data = np.concatenate((patch_data, patchdata), axis=0) if patch_data.size else patchdata
            patch_labels = np.concatenate((patch_labels, patchlabels), axis=0) if patch_labels.size else patchlabels
            patch_pws = np.concatenate((patch_pws, patchpw), axis=0) if patch_pws.size else patchpw
            patch_cmblabs = np.concatenate((patch_cmblabs, patchcmblabs),
                                           axis=0) if patch_cmblabs.size else patchcmblabs
            patch_cents.append(patchcents)

    return [patch_data, patch_labels, patch_pws, patch_cmblabs, patch_cents]


def fast_radial_symmetry_xfm(input_image, radii2d, alpha=2, factor_std=0.1, bright=False, dark=False):
    """
    :param input_image: input ndarray
    :param radii2d: list of integers
    :param alpha: input scalar
    :param factor_std: input float
    :param bright: bool
    :param dark: bool
    :return: dictionary of ndarrays
    """
    [gx, gy] = np.gradient(input_image)
    maximum_radius = np.ceil(np.max(radii2d))
    offset_img = np.array([maximum_radius, maximum_radius]).astype(int)
    rad_sym_output = np.zeros(input_image.shape + 2 * offset_img)

    Sum_sym = np.zeros([len(radii2d), rad_sym_output.shape[0], rad_sym_output.shape[1]])
    rad_index = 0
    for n in radii2d:
        O_n = np.zeros(rad_sym_output.shape)
        M_n = np.zeros(rad_sym_output.shape)
        for i in range(0, input_image.shape[0]):
            for j in range(0, input_image.shape[1]):
                p = np.array([i, j]).astype(int)
                g = np.array([gx[i, j], gy[i, j]]).astype(int)
                g_norm = np.sqrt(g @ g.T)
                if (g_norm > 0):
                    gp = np.round((g // g_norm) * n)
                    if bright:
                        ppos = p + gp
                        ppos = (ppos + offset_img)
                        O_n[int(ppos[0]), int(ppos[1])] = O_n[int(ppos[0]), int(ppos[1])] + 1
                        M_n[int(ppos[0]), int(ppos[1])] = M_n[int(ppos[0]), int(ppos[1])] + g_norm
                    if dark:
                        pneg = p - gp
                        pneg = (pneg + offset_img)
                        O_n[int(pneg[0]), int(pneg[1])] = O_n[int(pneg[0]), int(pneg[1])] - 1
                        M_n[int(pneg[0]), int(pneg[1])] = M_n[int(pneg[0]), int(pneg[1])] - g_norm

        O_n = abs(O_n)
        O_n = O_n / np.max(O_n)

        M_n = abs(M_n)
        M_n = M_n / np.max(M_n)

        S_n = (O_n ** alpha) * M_n

        Sum_sym[rad_index, :, :] = filters.gaussian_filter(S_n, n * factor_std)
        rad_index = rad_index + 1

    rad_sym_output = np.squeeze(np.sum(Sum_sym, axis=0))
    rad_sym_output = rad_sym_output[offset_img[0]:-offset_img[1], offset_img[0]:-offset_img[1]]
    return rad_sym_output


def get_frst_data(data):
    """
    :param data: input ndarray
    :return: frst ndarray
    """
    radii2d = np.array([2, 3])
    frst_output = np.zeros([data.shape[0], data.shape[1], data.shape[2]])
    for i in range(data.shape[2]):
        inp_slice = data[:, :, i]
        feature_frst = fast_radial_symmetry_xfm(inp_slice, radii2d, alpha=1, factor_std=0.1, bright=False,
                                                dark=True)
        frst_output[:, :, i] = feature_frst
    return frst_output


def vessel_detection2d(inp_images2d):
    """
    :param inp_images2d: input ndarray
    :return: list of ndarrays
    """
    im1 = np.copy(inp_images2d)
    frangi_output_volume = np.zeros([im1.shape[0], im1.shape[1], im1.shape[2]])
    label_image_volume = np.zeros([im1.shape[0], im1.shape[1], im1.shape[2]])
    newlabel_image_volume = np.zeros([im1.shape[0], im1.shape[1], im1.shape[2]])
    inpainted_result_volume = np.zeros([im1.shape[0], im1.shape[1], im1.shape[2]])
    for slice_no in range(0, im1.shape[2]):
        imslice = im1[:, :, slice_no]
        frangi_output = frangi(imslice, scale_range=(0.5, 1.2), scale_step=0.2, beta1=0.9, beta2=20,
                               black_ridges=True)  # (1, 3), 0.5, 100
        frangi_output_volume[:, :, slice_no] = frangi_output
        inp_feature = frangi_output.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(inp_feature)
        labels = kmeans.predict(inp_feature)
        labels = np.reshape(labels, (im1.shape[0], im1.shape[1]))
        label_img = label(labels)
        props = regionprops(label_img)
        label_img1 = np.copy(label_img)
        for index in range(0, len(props)):
            if props[index].eccentricity < 0.9 and props[index].solidity > 0.5:
                label_img1[label_img1 == index + 1] = 0
        newlabel = label(label_img1 > 0)
        label_image_volume[:, :, slice_no] = label_img
        newlabel_image_volume[:, :, slice_no] = newlabel
    return frangi_output_volume, newlabel_image_volume


def getting_candidates_from_frst(frst, img_resize):
    """
    :param frst: input ndarray
    :param img_resize: input ndarray
    :return: list of ndarrays
    """
    frst[np.isnan(frst)] = 0
    int95 = np.percentile(frst, 99.9)
    labfrst, nfrst = label(frst > int95, return_num=True)
    dist_labelled_frst = np.zeros_like(labfrst)
    dist_labelled_frst_vol = np.zeros_like(labfrst)
    av_distvals_frst = []
    av_distvals_frst_vol = []
    _, outlab = vessel_detection2d(img_resize)
    outlab = (outlab > 0).astype(float)
    inv_outlab = (1 - outlab).astype(float)
    for i in range(1, nfrst):
        sumval = np.sum(inv_outlab[labfrst == i])
        meanval = np.mean(inv_outlab[labfrst == i])
        dist_labelled_frst[labfrst == i] = sumval
        dist_labelled_frst_vol[labfrst == i] = meanval
        av_distvals_frst.append(meanval)
        av_distvals_frst_vol.append(sumval)
    print(np.array(av_distvals_frst))
    filtered_frst_cands = (dist_labelled_frst >= 12).astype(float) * (dist_labelled_frst_vol >= 1).astype(float)
    return filtered_frst_cands


def getting_cmb_data_disc_train(filename, patch_size=24):
    """
    :param filename: input list of directories
    :param patch_size: input scalar
    :return: list of ndarrays
    """
    outdir = '/path/to/the/output/directory'
    all_cmb_patches = np.array([])
    all_true_labels = np.array([])
    for im in range(len(filename)):
        img = nib.load('path/to/the/input_SWI_nifti.nii.gz').get_data().astype(
            float)
        # data = resize(data, [data.shape[0] // 2, data.shape[1] // 2, data.shape[2] // 2], preserve_range=True)
        try:
            gnd = nib.load('path/to/the/input_inplabel_nifti.nii.gz').get_data()
            # label = resize(label, [label.shape[0] // 2, label.shape[1] // 2, label.shape[2] // 2], preserve_range=True)
            gnd = (gnd > 0).astype(int)
        except:
            gnd = np.zeros_like(img, dtype=int)
        frst = np.load(filename[im])
        frst[np.isnan(frst)] = 0

        frst_cands = getting_candidates_from_frst(frst, img)
        prob_maps = np.load(outdir + 'prob_maps_only48_frst_allfolds_' + filename[im][60:-9] + '.npy')
        prob_maps[np.isnan(prob_maps)] = 0
        predicted = np.max(prob_maps, axis=0)
        predicted = predicted / np.amax(predicted)
        bin_map = (predicted > 0.1).astype(float) * (frst_cands > 0).astype(float)
        distprops = regionprops(label(bin_map[0] > 0))
        cents = np.zeros([len(distprops), 3])
        detected_cmbpatches_img = np.zeros([len(distprops), patch_size, patch_size, patch_size])
        detected_cmbpatches_frst = np.zeros([len(distprops), patch_size, patch_size, patch_size])
        true_labels = np.zeros([len(distprops), 2])
        ps = patch_size
        psz = patch_size
        for c in range(len(distprops)):
            cents[c, :] = distprops[c].centroid
            sx = np.amax([int(np.round(cents[c, 0])) - ps // 2, 0])
            ex = np.amin([int(np.round(cents[c, 0])) + ps // 2, bin_map.shape[0]])
            sy = np.amax([int(np.round(cents[c, 1])) - ps // 2, 0])
            ey = np.amin([int(np.round(cents[c, 1])) + ps // 2, bin_map.shape[1]])
            sz = np.amax([int(np.round(cents[c, 2])) - psz // 2, 0])
            ez = np.amin([int(np.round(cents[c, 2])) + psz // 2, bin_map.shape[2]])
            det_patch = img[sx:ex, sy:ey, sz:ez]
            det_patch_frst = frst[sx:ex, sy:ey, sz:ez]
            det_gnd_patch = gnd[sx:ex, sy:ey, sz:ez]
            if np.sum(det_gnd_patch) > 0:
                true_labels[c, 1] = 1
            else:
                true_labels[c, 0] = 1
            detected_cmbpatches_img[c, :det_patch.shape[0], :det_patch.shape[1], :det_patch.shape[2]] = det_patch
            detected_cmbpatches_frst[c, :det_patch.shape[0], :det_patch.shape[1], :det_patch.shape[2]] = det_patch_frst

        detected_cmbpatches_img = np.tile(detected_cmbpatches_img, (1, 1, 1, 1, 1))
        detected_cmbpatches_img = detected_cmbpatches_img.transpose(1, 2, 3, 4, 0)
        detected_cmbpatches_frst = np.tile(detected_cmbpatches_frst, (1, 1, 1, 1, 1))
        detected_cmbpatches_frst = detected_cmbpatches_frst.transpose(1, 2, 3, 4, 0)

        cmb_patches = np.concatenate([detected_cmbpatches_img, detected_cmbpatches_frst], axis=-1)
        all_cmb_patches = np.concatenate([all_cmb_patches, cmb_patches],
                                         axis=0) if all_cmb_patches.size else cmb_patches
        all_true_labels = np.concatenate([all_true_labels, true_labels],
                                         axis=0) if all_true_labels.size else true_labels
    return all_cmb_patches, all_true_labels


def getting_cmb_data_disc_testing(bin_map, testname, patch_size=24):
    """
    :param bin_map: input ndarray
    :param testname: input list of directories
    :param patch_size: input scalar
    :return: list of ndarrays
    """
    img = nib.load('path/to/the/input_SWI_nifti.nii.gz').get_data().astype(float)

    # img = resize(img, [img.shape[0] // 2, img.shape[1] // 2, img.shape[2] // 2], preserve_range=True)
    frst = np.load(testname[0])
    frst[np.isnan(frst)] = 0
    distprops = regionprops(label(bin_map > 0))
    cents = np.zeros([len(distprops), 3])
    detected_cmbpatches_img = np.zeros([len(distprops), patch_size, patch_size, patch_size])
    detected_cmbpatches_frst = np.zeros([len(distprops), patch_size, patch_size, patch_size])
    ps = patch_size
    psz = patch_size
    for c in range(len(distprops)):
        cents[c, :] = distprops[c].centroid
        sx = np.amax([int(np.round(cents[c, 0])) - ps // 2, 0])
        ex = np.amin([int(np.round(cents[c, 0])) + ps // 2, bin_map.shape[0]])
        sy = np.amax([int(np.round(cents[c, 1])) - ps // 2, 0])
        ey = np.amin([int(np.round(cents[c, 1])) + ps // 2, bin_map.shape[1]])
        sz = np.amax([int(np.round(cents[c, 2])) - psz // 2, 0])
        ez = np.amin([int(np.round(cents[c, 2])) + psz // 2, bin_map.shape[2]])
        det_patch = img[sx:ex, sy:ey, sz:ez]
        det_patch_frst = frst[sx:ex, sy:ey, sz:ez]

        detected_cmbpatches_img[c, :det_patch.shape[0], :det_patch.shape[1], :det_patch.shape[2]] = det_patch
        detected_cmbpatches_frst[c, :det_patch.shape[0], :det_patch.shape[1], :det_patch.shape[2]] = det_patch_frst

    detected_cmbpatches_img = np.tile(detected_cmbpatches_img, (1, 1, 1, 1, 1))
    detected_cmbpatches_img = detected_cmbpatches_img.transpose(1, 2, 3, 4, 0)

    detected_cmbpatches_frst = np.tile(detected_cmbpatches_frst, (1, 1, 1, 1, 1))
    detected_cmbpatches_frst = detected_cmbpatches_frst.transpose(1, 2, 3, 4, 0)

    cmb_patches = np.concatenate([detected_cmbpatches_img, detected_cmbpatches_frst], axis=-1)
    # cmb_patches = cmb_patches.transpose(0, 4, 1, 2, 3)

    return cmb_patches


def getting_cmb_data_stonline_train(filename, patch_size=24):
    """
    :param filename: input list of directories
    :param patch_size: input scalar
    :return: list of ndarrays
    """
    outdir = '/path/to/the/output/directory'
    all_cmb_patches = np.array([])
    all_true_labels = np.array([])
    all_gnd_patches = np.array([])
    for im in range(len(filename)):
        img = nib.load('path/to/the/input_SWI_nifti.nii.gz').get_data().astype(
            float)
        # data = resize(data, [data.shape[0] // 2, data.shape[1] // 2, data.shape[2] // 2], preserve_range=True)
        try:
            gnd = nib.load('path/to/the/input_inplabel_nifti.nii.gz').get_data()
            # label = resize(label, [label.shape[0] // 2, label.shape[1] // 2, label.shape[2] // 2], preserve_range=True)
            gnd = (gnd > 0).astype(int)
        except:
            gnd = np.zeros_like(img, dtype=int)
        frst = np.load(filename[im])
        frst[np.isnan(frst)] = 0

        frst_cands = getting_candidates_from_frst(frst, img)
        prob_maps = np.load(outdir + 'prob_maps_only48_frst_allfolds_' + filename[im][60:-9] + '.npy')
        prob_maps[np.isnan(prob_maps)] = 0
        predicted = np.max(prob_maps, axis=0)
        predicted = predicted / np.amax(predicted)
        bin_map = (predicted > 0.1).astype(float) * (frst_cands > 0).astype(float)
        distprops = regionprops(label(bin_map[0] > 0))
        cents = np.zeros([len(distprops), 3])
        detected_cmbpatches_img = np.zeros([len(distprops), patch_size, patch_size, patch_size])
        detected_cmbpatches_man = np.zeros([len(distprops), patch_size, patch_size, patch_size])
        detected_cmbpatches_frst = np.zeros([len(distprops), patch_size, patch_size, patch_size])
        true_labels = np.zeros([len(distprops), 2])
        ps = patch_size
        psz = patch_size
        for c in range(len(distprops)):
            cents[c, :] = distprops[c].centroid
            sx = np.amax([int(np.round(cents[c, 0])) - ps // 2, 0])
            ex = np.amin([int(np.round(cents[c, 0])) + ps // 2, bin_map.shape[0]])
            sy = np.amax([int(np.round(cents[c, 1])) - ps // 2, 0])
            ey = np.amin([int(np.round(cents[c, 1])) + ps // 2, bin_map.shape[1]])
            sz = np.amax([int(np.round(cents[c, 2])) - psz // 2, 0])
            ez = np.amin([int(np.round(cents[c, 2])) + psz // 2, bin_map.shape[2]])
            det_patch = img[sx:ex, sy:ey, sz:ez]
            det_patch_frst = frst[sx:ex, sy:ey, sz:ez]
            det_gnd_patch = gnd[sx:ex, sy:ey, sz:ez]
            if np.sum(det_gnd_patch) > 0:
                true_labels[c, 1] = 1
            else:
                true_labels[c, 0] = 1
            detected_cmbpatches_img[c, :det_patch.shape[0], :det_patch.shape[1], :det_patch.shape[2]] = det_patch
            detected_cmbpatches_man[c, :det_patch.shape[0], :det_patch.shape[1], :det_patch.shape[2]] = det_gnd_patch
            detected_cmbpatches_frst[c, :det_patch.shape[0], :det_patch.shape[1], :det_patch.shape[2]] = det_patch_frst

        detected_cmbpatches_img = np.tile(detected_cmbpatches_img, (1, 1, 1, 1, 1))
        detected_cmbpatches_img = detected_cmbpatches_img.transpose(1, 2, 3, 4, 0)
        detected_cmbpatches_frst = np.tile(detected_cmbpatches_frst, (1, 1, 1, 1, 1))
        detected_cmbpatches_frst = detected_cmbpatches_frst.transpose(1, 2, 3, 4, 0)

        cmb_patches = np.concatenate([detected_cmbpatches_img, detected_cmbpatches_frst], axis=-1)
        all_cmb_patches = np.concatenate([all_cmb_patches, cmb_patches],
                                         axis=0) if all_cmb_patches.size else cmb_patches
        all_gnd_patches = np.concatenate([all_gnd_patches, detected_cmbpatches_man],
                                         axis=0) if all_gnd_patches.size else detected_cmbpatches_man
        all_true_labels = np.concatenate([all_true_labels, true_labels],
                                         axis=0) if all_true_labels.size else true_labels
    return all_cmb_patches, all_gnd_patches, all_true_labels
