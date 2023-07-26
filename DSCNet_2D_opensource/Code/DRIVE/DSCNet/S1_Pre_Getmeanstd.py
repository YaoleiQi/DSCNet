# -*- coding: utf-8 -*-
from os import listdir
from os.path import join
import numpy as np
import SimpleITK as sitk

"""
The purpose of this code is to calculate the "mean" and "std" of the image, 
which will be used in the subsequent normalization process

Take the image ending with "nii.gz" as an example (using SimpleITK)
"""


def Getmeanstd(args, image_path, meanstd_name):
    """
    :param args: Parameters
    :param image_path: Address of image
    :param meanstd_name: save name of "mean" and "std"  (using ".npy" format to save)
    :return: None
    """
    root_dir = args.root_dir
    file_names = [x for x in listdir(join(image_path))]
    mean, std, length = 0.0, 0.0, 0.0

    for file_name in file_names:
        image = sitk.ReadImage(image_path + file_name)
        image = sitk.GetArrayFromImage(image).astype(np.float32)
        length += image.size
        mean += np.sum(image)
        # print(mean, length)
    mean = mean / length

    for file_name in file_names:
        image = sitk.ReadImage(image_path + file_name)
        image = sitk.GetArrayFromImage(image).astype(np.float32)
        std += np.sum(np.square((image - mean)))
        # print(std)
    std = np.sqrt(std / length)
    print("1 Finish Getmeanstd: ", meanstd_name)
    print("Mean and std are: ", mean, std)
    np.save(root_dir + meanstd_name, [mean, std])
