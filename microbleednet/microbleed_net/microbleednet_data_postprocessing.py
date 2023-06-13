from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from microbleednet.microbleed_net import microbleednet_data_preprocessing
import nibabel as nib
import glob
import random
from skimage.transform import resize
from skimage.measure import regionprops, label
from skimage.filters import frangi
from sklearn.cluster import KMeans
from scipy.ndimage import morphology, binary_fill_holes

#=========================================================================================
# Microbleednet data postprocessing function
# Vaanathi Sundaresan
# 10-01-2023
#=========================================================================================


def vessel_detection2d(inp_images2d, brain):

    im1 = np.copy(inp_images2d)
    frangi_output_volume = np.zeros([im1.shape[0], im1.shape[1], im1.shape[2]])
    label_image_volume = np.zeros([im1.shape[0], im1.shape[1], im1.shape[2]])
    newlabel_image_volume = np.zeros([im1.shape[0], im1.shape[1], im1.shape[2]])
    inpainted_result_volume = np.zeros([im1.shape[0], im1.shape[1], im1.shape[2]])
    for slice_no in range(0, im1.shape[2]):
        imslice = im1[:, :, slice_no]
        frangi_output = frangi(imslice, scale_range=(0.5, 1.2), scale_step=0.2, beta1=0.9, beta2=20, black_ridges=True) # (0.5, 1.2), 0.2, 0.9
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
                label_img1[label_img1 == index+1] = 0
        newlabel = label(label_img1 > 0)
        label_image_volume[:, :, slice_no] = label_img
        newlabel_image_volume[:, :, slice_no] = newlabel
    return frangi_output_volume, newlabel_image_volume


def remove_fps(pred, labmap):
    distmap = morphology.distance_transform_edt(labmap == 0)
    labout, nout = label(pred > 0, return_num=True)
    dist_labelled = np.zeros_like(labout)
    av_distvals = []
    for i in range(1, nout):
        meanval = np.mean(distmap[labout == i])
        dist_labelled[labout == i] = meanval
        av_distvals.append(meanval)
    dist_prob = (dist_labelled > 3).astype(float)
    return dist_prob


def postprocessing(output, img, thresh1, thresh2):
    brain_mask = img > 0
    brain_mask = binary_fill_holes(brain_mask)
    brain_mask = (morphology.distance_transform_edt(brain_mask > 0) > 8).astype(float)
    swi = img / np.amax(img)
    swi_thr = (swi < thresh1).astype(float)
    swi_thr_org = resize(swi_thr, [img.shape[0], img.shape[1], img.shape[2]], preserve_range=True)
    output_org = resize(output, [img.shape[0], img.shape[1], img.shape[2]], preserve_range=True)
    swi_thr = (swi_thr_org + (output_org > thresh2).astype(float)) * brain_mask.astype(float)
    labout, nout = label(swi_thr > 0, return_num=True)
    newlab = np.zeros_like(labout)
    for i in range(1, nout):
        newlab[labout == i] = np.sum((labout == i).astype(float))
    newlab[newlab > 125] = 0
    return newlab


def candidate_shapebased_filtering(heatmap3d, brain):
    brain_mask = (morphology.distance_transform_edt(brain > 0) > 8).astype(float)
    cd1_mask = heatmap3d > 0
    label3d, labelnum = label(cd1_mask, return_num=True)
    labchar_count = np.zeros([labelnum, 1])
    for slice in range(0, cd1_mask.shape[2]):
        cd1_mask_slice = cd1_mask[:, :, slice]
        label_img = label(cd1_mask_slice)
        prps = regionprops(label_img)
        for prp in prps:
            cent = prp.centroid
            labelid2 = label3d[int(cent[0]), int(cent[1])]
            if prp.eccentricity > 0.4 and prp.solidity < 0.5:
                labchar_count[labelid2 - 1] += 1
    tobezero_indices = np.where(labchar_count)[0]
    for tbz in range(len(tobezero_indices)):
        label3d[label3d == tobezero_indices[tbz] + 1] = 0

    labnew3d, nlabel = label(label3d > 0, return_num=True)
    for c in range(nlabel):
        ar = np.sum(labnew3d == c+1)
        if ar < 2:
            labnew3d[labnew3d == c+1] = 0

    return heatmap3d * (labnew3d > 0).astype(float) * brain_mask.astype(float)

