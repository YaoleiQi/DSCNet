# -*- coding: utf-8 -*-
import numpy as np
import SimpleITK as sitk
import random
from monai.transforms import (
    Orientationd,
    ScaleIntensityRanged,
    RandGaussianNoised,
    RandAffined,
    Rand2DElasticd,
    GaussianSmoothd,
)


def transform_img_lab(image, label, args):
    data_dicts = {"image": image, "label": label}
    orientation = Orientationd(keys=["image", "label"], as_closest_canonical=True)
    rand_affine = RandAffined(
        keys=["image", "label"],
        mode=("bilinear", "nearest"),
        prob=1.0,
        spatial_size=(args.ROI_shape, args.ROI_shape),
        translate_range=(30, 30),
        rotate_range=(np.pi / 36, np.pi / 36),
        scale_range=(0.15, 0.15),
        padding_mode="zeros",
    )
    rand_elastic = Rand2DElasticd(
        keys=["image", "label"],
        mode=("bilinear", "nearest"),
        prob=1.0,
        spacing=(20, 20),
        magnitude_range=(1, 1),
        spatial_size=(args.ROI_shape, args.ROI_shape),
        translate_range=(10, 20),
        rotate_range=(np.pi / 36, np.pi / 36),
        scale_range=(0.15, 0.15),
        padding_mode="zeros",
    )
    scale_shift = ScaleIntensityRanged(
        keys=["image"], a_min=-10, a_max=10, b_min=-10, b_max=10, clip=True
    )
    gauss_noise = RandGaussianNoised(keys=["image"], prob=1.0, mean=0.0, std=1)
    gauss_smooth = GaussianSmoothd(keys=["image"], sigma=0.6, approx="erf")

    # Params: Hyper parameters for data augumentation, number (like 0.3) refers to the possibility
    if random.random() > 0.2:
        if random.random() > 0.3:
            data_dicts = orientation(data_dicts)
        if random.random() > 0.3:
            if random.random() > 0.5:
                data_dicts = rand_affine(data_dicts)
            else:
                data_dicts = rand_elastic(data_dicts)
        if random.random() > 0.5:
            data_dicts = scale_shift(data_dicts)
        if random.random() > 0.5:
            if random.random() > 0.5:
                data_dicts = gauss_noise(data_dicts)
            else:
                data_dicts = gauss_smooth(data_dicts)
    else:
        data_dicts = data_dicts
    return data_dicts