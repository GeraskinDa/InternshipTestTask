import torch
import numpy as np
import os
import json
import sys
from matplotlib import pyplot as plt
from torchvision import transforms
from glob import glob
from PIL import Image, ImageEnhance

sys.path.append('..')
from lib import *


def save_masks(path):
    """Saving masks in jpg format from annotations in the folder
    path/masks.

    Parameters
    ----------
    path : str
        Path to the folder with annotations.

    Notes
    -----
    Corresponding folder should exist to save masks.

    """
    image_names = os.listdir(f'{path}/images')
    trans = transforms.ToPILImage()
    for img in image_names:
        img_id = int(img.split('.')[0])
        annotations = json.load(open(f'{path}/coco_annotations.json', 'r'))
        mask = trans(get_mask(img_id, annotations))
        mask.save(f"{path}/masks/m{img.split('.')[0]}.jpg")


def save_weight_masks(path):
    """Makes and saves weight masks in the folder path/weights.

    Parameters
    ----------
    path : str
        Path to the folder with the folder named 'masks'.

    Notes
    -----
    Corresponding folder should exist to save masks.

    """
    masks_names = os.listdir(f"{path}/masks")
    for mask in masks_names:
        mask_img = Image.open(f"{path}/masks/{mask}")
        weight_img = get_weight_mask(mask_img)
        weight_img.save(f"{path}/weights/w{mask[1:].split('.')[0]}.jpg")


def get_weight_mask(mask_img):
    """Makes a weight mask for the corresponding image.

    Parameters
    ----------
    mask_img : PIL Image format
        Mask image.

    Returns
    -------
    PIL Image format
        Weight mask in image format.

    """
    trans_to_tens = transforms.ToTensor()
    trans_to_pil = transforms.ToPILImage()
    mask_tens = trans_to_tens(mask_img)

    weight_tens = mask_tens.apply_(weights)

    weight_img = trans_to_pil(weight_tens)
    return weight_img


def weights(pix):
    """Function for element wise computing.

    Parameters
    ----------
    pix : int
        Value of the pixel.

    """
    if pix == 0:
        return 1
    else:
        return 5


def find_mean(path):
    """Finds mean value per channel of images.

    Parameters
    ----------
    path : str
        Path to the folder with images.

    Returns
    -------
    tensor, tensor.shape = [1, 3]
        Mean values from 0 to 1 per channel.

    """
    img_names = os.listdir(path)
    trans = transforms.ToTensor()
    img_num = len(img_names)
    sum = torch.Tensor([0, 0, 0])
    for idx in range(img_num):
        sum += torch.mean(trans(Image.open(f'{path}/{img_names[idx]}')),
                          dim=(1, 2))

    return sum / img_num


def find_std(path):
    """Finds std value per channel of images.

    Parameters
    ----------
    path : str
        Path to the folder with images.

    Returns
    -------
    tensor, tensor.shape = [1, 3]
        Std values from 0 to 1 per channel.

    """
    img_names = os.listdir(path)
    trans = transforms.ToTensor()
    img_num = len(img_names)
    sum = torch.Tensor([0, 0, 0])
    for idx in range(img_num):
        sum += torch.std(trans(Image.open(f'{path}/{img_names[idx]}')),
                         dim=(1, 2))

    return sum / img_num


def make_augment_and_save(path):
    """Makes augmented data and saves it in the folder path/augment/images,
    path/augment/masks and path/augment/weights.

    Parameters
    ----------
    path : str
        Path to the folder with folder 'images'.

    Notes
    -----
    Folders 'augment', 'images', 'masks' and 'weights' should exist.

    """
    sort_names = sort_by_cig(path)
    for i in range(4):
        start_idx = i * 500
        end_idx = (i + 1) * 500
        make_pics(path, sort_names[start_idx:end_idx], i + 1)


def make_pics(path, names, num):
    """Makes and saves images, masks and weight masks in the corresponding
    paths with num + 1 number cigarettes on it.
    For bigger cigarettes uses smaller num.

    Parameters
    ----------
    path : str
        Path to the folder.
    names : list[[str, str, str, int]]
        List with names of the images and value of pixels of cigarette.
    num : int from 1 to 4
        Desired number of cigarettes on the final image.

    """
    idx = 0
    if num == 1:
        for j in range(len(names)):
            if (idx + num) >= len(names):
                break
            img1 = Image.open(f"{path}/images/{names[idx][0]}")
            msk1 = Image.open(f"{path}/masks/{names[idx][1]}")
            wght1 = Image.open(f"{path}/weights/{names[idx][2]}")

            img1, msk1, wght1, idx, save = merge(path, names, idx,
                                                 img1, msk1, wght1)

            if save:
                save_augment_data(img1, msk1, wght1,
                                  path, names[idx][0])
            idx += 1

    if num == 2:
        for j in range(len(names)):
            if (idx + num) >= len(names):
                break
            img1 = Image.open(f"{path}/images/{names[idx][0]}")
            msk1 = Image.open(f"{path}/masks/{names[idx][1]}")
            wght1 = Image.open(f"{path}/weights/{names[idx][2]}")

            img1, msk1, wght1, idx, save = merge(path, names, idx,
                                                 img1, msk1, wght1)
            img1, msk1, wght1, idx, save = merge(path, names, idx,
                                                 img1, msk1, wght1)

            if save:
                save_augment_data(img1, msk1, wght1,
                                  path, names[idx][0])
            idx += 1

    if num == 3:
        for j in range(len(names)):
            if (idx + num) >= len(names):
                break
            img1 = Image.open(f"{path}/images/{names[idx][0]}")
            msk1 = Image.open(f"{path}/masks/{names[idx][1]}")
            wght1 = Image.open(f"{path}/weights/{names[idx][2]}")

            img1, msk1, wght1, idx, save = merge(path, names, idx,
                                                 img1, msk1, wght1)
            img1, msk1, wght1, idx, save = merge(path, names, idx,
                                                 img1, msk1, wght1)
            img1, msk1, wght1, idx, save = merge(path, names, idx,
                                                 img1, msk1, wght1)

            if save:
                save_augment_data(img1, msk1, wght1,
                                  path, names[idx][0])
            idx += 1

    if num == 4:
        for j in range(len(names)):
            if (idx + num) >= len(names):
                break
            img1 = Image.open(f"{path}/images/{names[idx][0]}")
            msk1 = Image.open(f"{path}/masks/{names[idx][1]}")
            wght1 = Image.open(f"{path}/weights/{names[idx][2]}")

            img1, msk1, wght1, idx, save = merge(path, names, idx,
                                                 img1, msk1, wght1)
            img1, msk1, wght1, idx, save = merge(path, names, idx,
                                                 img1, msk1, wght1)
            img1, msk1, wght1, idx, save = merge(path, names, idx,
                                                 img1, msk1, wght1)
            img1, msk1, wght1, idx, save = merge(path, names, idx,
                                                 img1, msk1, wght1)

            if save:
                save_augment_data(img1, msk1, wght1,
                                  path, names[idx][0])
            idx += 1


def merge(path, names, idx, img1, msk1, wght1):
    """Merges two images with cigarettes, if it is possible:
    they don't overlap.

    Parameters
    ----------
    path : str
        Path to the folder.
    names : list[[str, str, str, int]]
        List with names of the images and value of pixels of cigarette.
    idx : int
        Index of the current position in names.
    img1 : PIL Image format
        Image with the name names[idx][0].
    msk1 : PIL Image format
        Mask with the name names[idx][1].
    wght1 : PIL Image format
        Weight mask with the name names[idx][2].

    Returns
    -------
    img1 : PIL Image format
        Merged images.
    msk1 : PIL Image format
        Merged masks.
    wght1 : PIL Image format
        Merged weights.
    idx : int
        New index of the current position in names.
    save : bool
        Shows if the result should be saved.

    """
    trans_to_tens = transforms.ToTensor()
    trans_to_pil = transforms.ToPILImage()
    msk2 = Image.open(f"{path}/masks/{names[idx + 1][1]}")
    save = False
    angle = 0
    for i in range(4):
        angle = i * 90
        new = msk2.rotate(angle)
        if is_merge(trans_to_tens(msk1), trans_to_tens(new)):
            save = True
            break
    if save:
        msk2 = msk2.rotate(angle)
        msk2_tens = trans_to_tens(msk2)
        msk2_tens_unsq = make_three_chan(msk2_tens)
        msk_final = trans_to_pil(trans_to_tens(msk1)
                                 + msk2_tens)

        wght2 = Image.open(f"{path}/weights/"
                           f"{names[idx + 1][2]}").rotate(angle)
        wght_final = trans_to_pil(trans_to_tens(wght1)
                                  + trans_to_tens(wght2))

        img2 = Image.open(f"{path}/images/"
                          f"{names[idx + 1][0]}").rotate(angle)
        cig2 = trans_to_tens(img2) * msk2_tens_unsq

        img_final = trans_to_pil((1 - msk2_tens_unsq)
                                 * trans_to_tens(img1) + cig2)

        idx += 1
        return img_final, msk_final, wght_final, idx, save

    else:
        idx += 1
        return img1, msk1, wght1, idx, save


def is_merge(mask1, mask2):
    """Checks if two images can be merged.

    Parameters
    ----------
    mask1 : tensor, tensor.shape = [1, 512, 512]
        Tensor of the first mask.
    mask2 : tensor, tensor.shape = [1, 512, 512]
        Tensor of the second mask.
    Returns
    -------
    bool
        If true, two images can be merged.
    """
    ans = torch.sum((mask1+mask2) == 2)
    if ans > 0:
        return False
    else:
        return True


def sort_by_cig(path):
    """Sorts the list of names, masks and weights by the number of pixels
    corresponding to the cigarette.

    Parameters
    ----------
    path : str
        Path to the folder with folders 'images', 'masks', 'weights'.

    Returns
    -------
    list[[str, str, str, int]]
        Sorted by the number of pixels corresponding to the cigarette list (reversed).

    """
    img_names = sorted(os.listdir(f"{path}/images"))
    mask_names = sorted(os.listdir(f"{path}/masks"))
    weight_names = sorted(os.listdir(f"{path}/weights"))

    cig_pxl = cig_pxl_sum(path, mask_names)
    pack = [[i, j, k, m] for i, j, k, m in zip(img_names,
                                               mask_names,
                                               weight_names, cig_pxl)]

    sort_by_cig_lst = sorted(pack, key=lambda pxl: pxl[3], reverse=True)

    return sort_by_cig_lst


def cig_pxl_sum(path, mask_names):
    """Computes the number of pixels corresponding to the cigarette on the masks.

    Parameters
    ----------
    path : str
        Path to the folder with masks.

    Returns
    -------
    list[int]
        List of the number of pixels.

    """
    trans_to_tens = transforms.ToTensor()
    cig_pxl = []
    for i in range(len(mask_names)):
        mask = Image.open(f"{path}/masks/{mask_names[i]}")
        cig_pxl.append(torch.sum(trans_to_tens(mask)))

    return cig_pxl


def save_augment_data(img, mask, weight, path, name):
    """Saves images, masks and weights with name 'name' in the folders
    path/augment/images, path/augment/masks and path/augment/weights.

    Parameters
    ----------
    img : PIL Image format
        Image that should be saved.
    mask : PIL Image format
        Mask that should be saved.
    weight : PIL Image format
        Weight that should be saved.    

    """
    img.save(f"{path}/augment/images/aug{name}")
    mask.save(f"{path}/augment/masks/aug_m{name}")
    weight.save(f"{path}/augment/weights/aug_w{name}")


def make_three_chan(mask):
    return torch.cat((mask, mask, mask))


# Make before the start
# make_augment_and_save('../data/cig_butts/train')
# mean = find_mean('../data/cig_butts/train/images')
# print(f"Mean per channel are {mean}")
# std = find_std('../data/cig_butts/train/images')
# print(f"Std per channel are {std}")

# Making in the main body

# path = '../data/cig_butts/train'
# save_masks(path)

# path = '../data/cig_butts/val'
# save_masks(path)

# path = '../data/cig_butts/train'
# save_weight_masks(path)

# path = '../data/cig_butts/val'
# save_weight_masks(path)




