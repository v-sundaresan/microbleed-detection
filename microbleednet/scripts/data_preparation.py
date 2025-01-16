from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import nibabel as nib
from tqdm import tqdm
from skimage.filters import frangi
from sklearn.cluster import KMeans
from skimage.morphology import erosion, ball
from skimage.measure import regionprops, label
from microbleednet.scripts import augmentations
from scipy.ndimage import filters, convolve, distance_transform_edt
from skimage.feature import structure_tensor, structure_tensor_eigenvalues


###########################################
# Microbleednet data preparation function #
# Vaanathi Sundaresan                     #
# 10-01-2023                              #
###########################################

def find_bounds_1d(array):
    """
    Finds the bounds of non-zero values in a 1D array.

    Parameters:
    array : ndarray
        A 1D array of numeric values.

    Returns:
    start_index : int
        The starting index of the non-zero region.
    end_index : int
        The ending index of the non-zero region (inclusive).
    length : int
        The length of the non-zero region.
    """
    nonzero_mask = list(array > 0)
    start_index = nonzero_mask.index(1)
    reverse_start_index = nonzero_mask[::-1].index(1)
    length = len(array[start_index:]) - reverse_start_index

    return start_index, reverse_start_index, length

def tight_crop(image):
    """
    Crops a 3D image array to remove zero-value regions.

    Parameters:
    image : ndarray
        A 3D NumPy array representing the input image.

    Returns:
    cropped_image : ndarray
        The cropped image containing non-zero regions.
    bounding_box : list of int
        Bounding box as [row_start, row_length, col_start, col_length, stack_start, stack_length].
    """

    row_sum = np.sum(np.sum(image, axis=1), axis=1)
    col_sum = np.sum(np.sum(image, axis=0), axis=1)
    stack_sum = np.sum(np.sum(image, axis=1), axis=0)

    rsid, reid, rlen = find_bounds_1d(row_sum)
    csid, ceid, clen = find_bounds_1d(col_sum)
    ssid, seid, slen = find_bounds_1d(stack_sum)

    return image[rsid:rsid+rlen, csid:csid+clen, ssid:ssid+slen], [rsid, rlen, csid, clen, ssid, slen]

def scale_data(data):
    # return (2 * (data / np.amax(np.reshape(data, [-1, 1])))) - 1
    return 2 * (data / np.max(data)) - 1

def invert_data(data):
    # data1 = 1 - (data / np.amax(np.reshape(data, [-1, 1])))
    return 1 - (data / np.max(data))

def inpaint_with_neighborhood_mean(image, mask):
    
    '''Inpaint point with mean of 26-connected neighbourhood'''

    inpainted_volume = image.copy()
    
    kernel = np.ones((3, 3, 3))
    kernel[1, 1, 1] = 0

    with tqdm(leave=False, desc='inpainting_mask', disable=True) as pbar:
        while mask.sum() > 0:
            neighbour_sum = convolve(inpainted_volume * (1 - mask), kernel, mode='constant', cval=0)
            neighbour_count = convolve(1 - mask, kernel, mode='constant', cval=0)

            mean_values = np.zeros_like(neighbour_sum)  # Initialize with zeros
            valid_neighbours = neighbour_count > 0
            mean_values[valid_neighbours] = neighbour_sum[valid_neighbours] / neighbour_count[valid_neighbours]

            inpainted_volume[mask == 1] = mean_values[mask == 1]
            mask[mask == 1] = neighbour_count[mask == 1] == 0
            pbar.update()

    return inpainted_volume

def eigenvalues_and_linearity_measure(image_slice):

    Ixx, Ixy, Iyy = structure_tensor(image_slice)
    eigenvals = structure_tensor_eigenvalues((Ixx, Ixy, Iyy))

    lambda1 = eigenvals[0]
    lambda2 = eigenvals[1]
    linearity_measure = np.absolute((lambda1 - lambda2) / 2)
    return lambda1, lambda2, linearity_measure

def inpaint_vessels(image, brain_mask):

    labelled_mask_volume = np.zeros_like(image)
    inpainted_volume = np.zeros_like(image)

    height, width, depth = image.shape

    for slice_idx in tqdm(range(depth), leave=False, desc='generating_vessel_masks', disable=True):

        slice = image[:, :, slice_idx]

        if np.min(slice) == np.max(slice):
            # slice is empty
            continue

        frangi_output_slice = frangi(slice, sigmas=(0.5, 1.2, 0.2), alpha=0.9, beta=20, black_ridges=True)
        frangi_output_slice = frangi_output_slice * brain_mask[:, :, slice_idx].astype(float)

        _, _, linearity = eigenvalues_and_linearity_measure(slice)
        linearity = linearity * brain_mask[:, :, slice_idx].astype(float)
        linearity -= np.min(linearity)
        linearity = linearity / np.max(linearity) if np.max(linearity) != 0 else linearity

        slice_features = np.stack([frangi_output_slice.ravel(), linearity.ravel()], axis=1)
        # slice_features = np.reshape(frangi_output_slice, (-1, 1))

        clusterer = KMeans(n_clusters=2, random_state=42).fit(slice_features)
        clusters = clusterer.predict(slice_features)

        # Assuming that the number of pixels in vessels is less than other pixels, we label clusters
        vessel_cluster_label = 1 if (clusters == 1).sum() < (clusters == 0).sum() else 0
        
        vessel_mask = np.reshape(clusters == vessel_cluster_label, (height, width))
        # vessel_mask = np.reshape(mask_features, (height, width))

        labelled_mask = label(vessel_mask)
        mask_props = regionprops(labelled_mask)

        for mask_prop in mask_props:
            if mask_prop.eccentricity < 0.9 and mask_prop.solidity > 0.5:
                labelled_mask[labelled_mask == mask_prop.label] = 0

        labelled_mask_volume[:, :, slice_idx] = labelled_mask

    labelled_mask_volume = (labelled_mask_volume > 0).astype(float)
    inpainted_volume = inpaint_with_neighborhood_mean(image, labelled_mask_volume)

    return inpainted_volume

def preprocess_subject(subject):
    
    # Load image
    image_path = subject['input_path']
    image = nib.load(image_path).get_fdata()

    # Inpaint vessels
    brain_mask = (image > 0).astype(int)
    image = inpaint_vessels(image, brain_mask)

    # Load label
    try:
        label_path = subject['label_path']
        label = label = nib.load(label_path).get_fdata()
        label = (label > 0).astype(int)
    except:
        label = np.zeros_like(image, dtype=int)

    # Load FRST
    try:
        frst_path = subject['frst_path']
        frst = nib.load(frst_path).get_fdata()
    except:
        frst = get_frst_data(image)
        frst[np.isnan(frst)] = 0
        frst -= np.min(frst)
        frst /= np.max(frst)

    # Invert image
    image = 1 - (image / np.max(image))
    image = image * brain_mask.astype(float)

    return image, label, frst

def load_subject(subject):

    # Load image
    image_path = subject['input_path']
    image = nib.load(image_path).get_fdata()

    brain_mask = (image > 0).astype(int)
    # image = invert_data(image)
    # image = scale_data(image)

    image = image * brain_mask.astype(float)

    # Load label
    try:
        label_path = subject['label_path']
        label = label = nib.load(label_path).get_fdata()
        label = (label > 0).astype(int)
    except:
        label = np.zeros_like(image, dtype=int)

    # Load FRST
    try:
        frst_path = subject['frst_path']
        frst = nib.load(frst_path).get_fdata()
    except:
        frst = get_frst_data(image)
        frst[np.isnan(frst)] = 0
        frst -= np.min(frst)
        frst /= np.max(frst)

    frst = frst * brain_mask.astype(float)

    # crop all arrays
    _, coords = tight_crop(image)
    frst = frst[coords[0]:coords[0] + coords[1], coords[2]:coords[2] + coords[3], coords[4]:coords[4] + coords[5]]
    label = label[coords[0]:coords[0] + coords[1], coords[2]:coords[2] + coords[3], coords[4]:coords[4] + coords[5]]
    image = image[coords[0]:coords[0] + coords[1], coords[2]:coords[2] + coords[3], coords[4]:coords[4] + coords[5]]

    return image, label, frst, coords

def get_nonoverlapping_patches(image, label, brain_mask, frst=None, patch_size=32):
    
    brain_mask = erosion(brain_mask, ball(2))
        
    image = image * brain_mask
    label = (label > 0).astype(int)

    num_patches_x = np.ceil(image.shape[0] / patch_size).astype(int)
    num_patches_y = np.ceil(image.shape[1] / patch_size).astype(int)
    num_patches_z = np.ceil(image.shape[2] / patch_size).astype(int)
    num_patches = num_patches_x * num_patches_y * num_patches_z

    image_patches = np.zeros([num_patches, patch_size, patch_size, patch_size])
    label_patches = np.zeros([num_patches, patch_size, patch_size, patch_size])

    if frst is not None:
        frst_patches = np.zeros([num_patches, patch_size, patch_size, patch_size])

    patch_idx = 0
    patch_labels = np.zeros([num_patches, 2])
    patch_centroids = np.zeros([num_patches, 3])

    for z in range(num_patches_z):
        for y in range(num_patches_y):
            for x in tqdm(range(num_patches_x), leave=False, desc='generating_nonoverlapping_patches', disable=True):

                patch_x_start = x * patch_size
                patch_x_end = min((x + 1) * patch_size, image.shape[0])
                if x == num_patches_x - 1:
                    patch_x_start = max(0, image.shape[0] - patch_size)

                patch_y_start = y * patch_size
                patch_y_end = min((y + 1) * patch_size, image.shape[1])
                if y == num_patches_y - 1:
                    patch_y_start = max(0, image.shape[1] - patch_size)

                patch_z_start = z * patch_size
                patch_z_end = min((z + 1) * patch_size, image.shape[2])
                if z == num_patches_z - 1:
                    patch_z_start = max(0, image.shape[2] - patch_size)

                image_patch = image[patch_x_start:patch_x_end, patch_y_start:patch_y_end, patch_z_start:patch_z_end]
                label_patch = label[patch_x_start:patch_x_end, patch_y_start:patch_y_end, patch_z_start:patch_z_end]
                patch_centroids[patch_idx, ...] = [(patch_x_end - patch_x_start) / 2, (patch_y_end - patch_y_start) / 2, (patch_z_end - patch_z_start) / 2]

                image_patches[patch_idx, :image_patch.shape[0], :image_patch.shape[1], :image_patch.shape[2]] = image_patch
                label_patches[patch_idx, :label_patch.shape[0], :label_patch.shape[1], :label_patch.shape[2]] = label_patch

                if np.sum(label_patch) > 0:
                    # If the patch contains a labelled CMB
                    patch_labels[patch_idx, 1] = 1
                else:
                    patch_labels[patch_idx, 0] = 1

                if frst is not None:
                    frst_patch = frst[patch_x_start:patch_x_end, patch_y_start:patch_y_end, patch_z_start:patch_z_end]
                    frst_patches[patch_idx, :frst_patch.shape[0], :frst_patch.shape[1], :frst_patch.shape[2]] = frst_patch

                patch_idx += 1

    if frst is not None:
        image_patches = np.stack([image_patches, frst_patches], axis=1)

    smoothed_label_patches = filters.gaussian_filter(label_patches, 1.2) * 10

    return image_patches, label_patches, smoothed_label_patches, patch_labels, patch_centroids

def augment_data(data, labels, n_augmentations=4):
    
    n_samples = data.shape[0]

    augmented_frst_store = []
    augmented_image_store = []
    augmented_label_store = []

    for idx in range(n_samples):
        for _ in range(n_augmentations):

            frst = data[idx, 1]
            image = data[idx, 0]
            label = labels[idx]

            augmented_image, augmented_frst, augmented_label = augmentations.augment_pair(image, frst, label)

            augmented_frst_store.append(augmented_frst)
            augmented_image_store.append(augmented_image)
            augmented_label_store.append(augmented_label)

    # Convert list to array
    augmented_frsts = np.array(augmented_frst_store)
    augmented_images = np.array(augmented_image_store)
    augmented_labels = np.array(augmented_label_store)

    # Stack images and FRST
    augmented_data = np.stack((augmented_images, augmented_frsts), axis=1)

    # Add augmentations to original data
    augmented_data = np.concatenate((data, augmented_data), axis=0)
    augmented_labels = np.concatenate((labels, augmented_labels), axis=0)

    return augmented_data, augmented_labels

def split_into_nonoverlapping_patches_classwise(subjects, patch_size=48):

    positive_patches_store = []
    negative_patches_store = []

    for subject in tqdm(subjects, leave=False, desc='split_into_nonoverlapping_patches_classwise', disable=True):

        image, label, frst, _ = load_subject(subject)
        brain_mask = (image > 0).astype(int)

        data_patches, label_patches, patch_pixel_weights, patch_labels, _ = get_nonoverlapping_patches(image, label, brain_mask, frst, patch_size)
        data_patches[data_patches < 0] = 0

        n_patches = data_patches.shape[0]
        for patch_idx in range(n_patches):

            instance = {
                    'data_patch': data_patches[patch_idx],
                    'label_patch': label_patches[patch_idx],
                    'patch_pixel_weights': patch_pixel_weights[patch_idx],
                }

            if patch_labels[patch_idx, 1] == 1:
                positive_patches_store.append(instance)
            else:
                negative_patches_store.append(instance)

    return positive_patches_store, negative_patches_store

def put_patches_into_volume(patches, volume_template, patch_size):

    volume = np.zeros_like(volume_template)

    num_patches_x = np.ceil(volume.shape[0] / patch_size).astype(int)
    num_patches_y = np.ceil(volume.shape[1] / patch_size).astype(int)
    num_patches_z = np.ceil(volume.shape[2] / patch_size).astype(int)

    patch_idx = 0
    for z in range(num_patches_z):
        for y in range(num_patches_y):
            for x in tqdm(range(num_patches_x), leave=False, desc='replacing_patches', disable=True):

                patch_x_start = x * patch_size
                patch_x_end = min((x + 1) * patch_size, volume.shape[0])
                if x == num_patches_x - 1:
                    patch_x_start = max(0, volume.shape[0] - patch_size)

                patch_y_start = y * patch_size
                patch_y_end = min((y + 1) * patch_size, volume.shape[1])
                if y == num_patches_y - 1:
                    patch_y_start = max(0, volume.shape[1] - patch_size)

                patch_z_start = z * patch_size
                patch_z_end = min((z + 1) * patch_size, volume.shape[2])
                if z == num_patches_z - 1:
                    patch_z_start = max(0, volume.shape[2] - patch_size)

                volume[patch_x_start:patch_x_end, patch_y_start:patch_y_end, patch_z_start:patch_z_end] = patches[patch_idx, :(patch_x_end-patch_x_start), :(patch_y_end-patch_y_start), :(patch_z_end-patch_z_start)]


                patch_idx += 1
    
    return volume

def get_patches_centered_on_cmb(image, gt, frst=None, patch_size=24):
    
    # label is called gt here because of the conflict with the function 'label'
    labelled_gt = label(gt)
    dist_properties = regionprops(labelled_gt)
    n_patches = len(dist_properties)

    patch_labels = np.zeros([n_patches, 2])
    frst_patches = np.zeros([n_patches, patch_size, patch_size, patch_size])
    image_patches = np.zeros([n_patches, patch_size, patch_size, patch_size])

    for patch_idx in tqdm(range(n_patches), leave=False, desc='get_patches_centered_on_cmb', disable=True):

        centroid = dist_properties[patch_idx].centroid
        sx = max(0, int(np.round(centroid[0])) - patch_size // 2)
        ex = min(gt.shape[0], int(np.round(centroid[0])) + patch_size // 2)
        sy = max(0, int(np.round(centroid[1])) - patch_size // 2)
        ey = min(gt.shape[1], int(np.round(centroid[1])) + patch_size // 2)
        sz = max(0, int(np.round(centroid[2])) - patch_size // 2)
        ez = min(gt.shape[2], int(np.round(centroid[2])) + patch_size // 2)

        label_patch = gt[sx:ex, sy:ey, sz:ez]
        frst_patch = frst[sx:ex, sy:ey, sz:ez]
        image_patch = image[sx:ex, sy:ey, sz:ez]

        component_mask = (labelled_gt[sx:ex, sy:ey, sz:ez] == dist_properties[patch_idx].label)
        isolated_component = label_patch * component_mask

        if 2 in isolated_component:
            # True positive patch
            patch_labels[patch_idx, 1] = 1
        else:
            # False positive patch
            patch_labels[patch_idx, 0] = 1

        frst_patches[patch_idx, :frst_patch.shape[0], :frst_patch.shape[1], :frst_patch.shape[2]] = frst_patch
        image_patches[patch_idx, :image_patch.shape[0], :image_patch.shape[1], :image_patch.shape[2]] = image_patch

    image_patches = np.stack([image_patches, frst_patches], axis=1)

    return image_patches, patch_labels

def split_into_patches_centered_on_cmb_classwise(subjects, patch_size=24):

    tp_patches_store = []
    fp_patches_store = []

    for subject in tqdm(subjects, leave=False, desc='split_into_patches_centered_on_cmb_classwise', disable=True):

        image_header = nib.load(subject['input_path']).header
        image, label, frst, _ = load_subject(subject)
        cdet_prediction = subject['cdet_inference']

        # Here, if it is true positive, the detected cmb is labelled as 2, else it is labelled as 1
        label = np.minimum((label * 2) + cdet_prediction, 2)

        data_patches, patch_labels = get_patches_centered_on_cmb(image, label, frst, patch_size)
        data_patches[data_patches < 0] = 0

        n_patches = data_patches.shape[0]
        for patch_idx in range(n_patches):

            instance = {
                    'data_patch': data_patches[patch_idx],
                    'patch_label': patch_labels[patch_idx],
                }
            
            if patch_labels[patch_idx, 1] == 1:
                tp_patches_store.append(instance)
            else:
                fp_patches_store.append(instance)
    
    return tp_patches_store, fp_patches_store

def filter_predictions_from_volume(prediction_volume, component_labels):

    labelled_prediction_volume = label(prediction_volume)
    dist_properties = regionprops(labelled_prediction_volume)
    n_patches = len(dist_properties)

    for patch_idx in tqdm(range(n_patches), leave=False, desc='filter_predictions_from_volume', disable=True):

        component_label = component_labels[patch_idx]
        prediction_volume[labelled_prediction_volume == dist_properties[patch_idx].label] = component_label

    return prediction_volume

def shape_based_filtering(prediction_volume, brain_mask):
    
    brain_mask = (distance_transform_edt(brain_mask) > 8).astype(float)
    labelled_prediction_volume = label(prediction_volume)
    
    height, width, depth = prediction_volume.shape

    for slice_idx in range(depth):
        slice_labels = labelled_prediction_volume[:, :, slice_idx]
        slice_props = regionprops(slice_labels)

        for region in slice_props:
            if region.eccentricity > 0.4 and region.solidity < 0.5:
                labelled_prediction_volume[labelled_prediction_volume == region.label] = 0

    labelled_prediction_volume, n_predictions = label(labelled_prediction_volume > 0, return_num=True)
    for label_id in range(1, n_predictions + 1):
        region = (labelled_prediction_volume == label_id)
        region_size = np.sum(region)
        if region_size < 2:
            labelled_prediction_volume[region] = 0

    prediction_volume = (labelled_prediction_volume > 0).astype(float)
    prediction_volume = prediction_volume * brain_mask

    return prediction_volume

def fast_radial_symmetry_xfm(image, radii, alpha=2, factor_std=0.1, bright=False, dark=False):
    
    np.seterr(invalid='ignore')
    
    [gx, gy] = np.gradient(image)
    maximum_radius = np.ceil(np.max(radii))
    offset_img = np.array([maximum_radius, maximum_radius]).astype(int)
    rad_sym_output = np.zeros(image.shape + 2 * offset_img)

    rad_index = 0
    Sum_sym = np.zeros([len(radii), rad_sym_output.shape[0], rad_sym_output.shape[1]])

    for n in radii:
        O_n = np.zeros(rad_sym_output.shape)
        M_n = np.zeros(rad_sym_output.shape)
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
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

    radii = np.array([2, 3])
    frst = np.zeros_like(data)
    height, width, depth = data.shape

    for idx in tqdm(range(depth), leave=False, desc='get_frst_data'):
        slice = data[:, :, idx]
        frst[:, :, idx] = fast_radial_symmetry_xfm(slice, radii, alpha=2, factor_std=0.1, bright=False, dark=True)

    return frst
