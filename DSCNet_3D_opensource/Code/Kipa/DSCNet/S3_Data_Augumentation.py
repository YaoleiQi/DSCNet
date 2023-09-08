# -*- coding: utf-8 -*-
import numpy as np
import SimpleITK as sitk
import random
from monai.transforms import (
    Orientationd,
    ScaleIntensityRanged,
    RandGaussianNoised,
    RandAffined,
    Rand3DElasticd,
    GaussianSmoothd,
)


def transform_img_lab(image, label, args):
    data_dicts = {"image": image, "label": label}
    orientation = Orientationd(keys=["image", "label"], axcodes="RSA")
    rand_affine = RandAffined(
        keys=["image", "label"],
        mode=("bilinear", "nearest"),
        prob=1.0,
        spatial_size=args.ROI_shape,
        translate_range=(2, 30, 30),
        rotate_range=(np.pi / 4, np.pi / 36, np.pi / 36),
        scale_range=(0.15, 0.15, 0.15),
        padding_mode="zeros")
    rand_elastic = Rand3DElasticd(
        keys=["image", "label"],
        mode=("bilinear", "nearest"),
        prob=1.0,
        sigma_range=(5, 8),
        magnitude_range=(100, 200),
        spatial_size=args.ROI_shape,
        translate_range=(2, 30, 30),
        rotate_range=(np.pi/4, np.pi / 36, np.pi/36),
        scale_range=(0.15, 0.15, 0.15),
        padding_mode="zeros")
    scale_shift = ScaleIntensityRanged(
        keys=["image"], a_min=-10, a_max=10, b_min=-10, b_max=10, clip=True
    )
    gauss_noise = RandGaussianNoised(keys=["image"], prob=1.0, mean=0.0, std=1)
    gauss_smooth = GaussianSmoothd(keys=["image"], sigma=0.6, approx="erf")

    # Params: Hyper parameters for data augumentation, number (like 0.3) refers to the possibility
    if random.random() > 0.5:
        if random.random() > 0.5:
            data_dicts = orientation(data_dicts)
        if random.random() > 0.4:
            if random.random() > 0.7:
                data_dicts = rand_affine(data_dicts)
            else:
                data_dicts = rand_elastic(data_dicts)
        if random.random() > 0.7:
            data_dicts = scale_shift(data_dicts)
        if random.random() > 0.8:
            if random.random() > 0.7:
                data_dicts = gauss_noise(data_dicts)
            else:
                data_dicts = gauss_smooth(data_dicts)
    else:
        data_dicts = data_dicts
    return data_dicts