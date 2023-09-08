# -*- coding: utf-8 -*-
import torch
from torch.utils import data
import numpy as np
import SimpleITK as sitk
from S3_Data_Augumentation import transform_img_lab
import warnings
warnings.filterwarnings("ignore")


def read_file_from_txt(txt_path):
    files = []
    for line in open(txt_path, "r"):
        files.append(line.strip())
    return files


def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape

    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((num_classes, n))
    categorical[y, np.arange(n)] = 1

    output_shape = (num_classes,) + input_shape
    categorical = np.reshape(categorical, output_shape)
    return categorical


def reshape_img(image, z, y, x):
    out = np.zeros([z, y, x], dtype=np.float32)
    out[0:image.shape[0], 0:image.shape[1], 0:image.shape[2]] = image[0:image.shape[0], 0:image.shape[1],
                                                                0:image.shape[2]]
    return out


class Dataloader(data.Dataset):
    def __init__(self, args):
        super(Dataloader, self).__init__()
        self.image_file = read_file_from_txt(args.Image_Tr_txt)
        self.label_file = read_file_from_txt(args.Label_Tr_txt)
        self.shape = args.ROI_shape
        self.args = args

    def __getitem__(self, index):
        image_path = self.image_file[index]
        label_path = self.label_file[index]

        # Read images and labels
        image = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image)
        label = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(label)

        z, y, x = image.shape
        image = image.astype(dtype=np.float32)
        label = label.astype(dtype=np.float32)

        # Normalization
        mean, std = np.load(self.args.root_dir + self.args.Tr_Meanstd_name)
        image = (image - mean) / std

        if self.shape[0] > z:
            z = self.shape[0]
            image = reshape_img(image, z, y, x)
            label = reshape_img(label, z, y, x)
        if self.shape[1] > y:
            y = self.shape[1]
            image = reshape_img(image, z, y, x)
            label = reshape_img(label, z, y, x)
        if self.shape[2] > x:
            x = self.shape[2]
            image = reshape_img(image, z, y, x)
            label = reshape_img(label, z, y, x)

        # Random crop, (center_y, center_x) refers the left-up coordinate of the Random_Crop_Block
        center_z = np.random.randint(0, z - self.shape[0] + 1, 1, dtype=np.int16)[0]
        center_y = np.random.randint(0, y - self.shape[1] + 1, 1, dtype=np.int16)[0]
        center_x = np.random.randint(0, x - self.shape[2] + 1, 1, dtype=np.int16)[0]
        image = image[center_z:self.shape[0] +
                               center_z, center_y:self.shape[1] + center_y, center_x:self.shape[2] + center_x]
        label = label[center_z:self.shape[0] +
                                 center_z, center_y:self.shape[1] + center_y, center_x:self.shape[2] + center_x]

        image = image[np.newaxis, :, :, :]  # b c d w h
        label = label[np.newaxis, :, :, :]  # b c d w h

        # Data Augmentation
        data_dict = transform_img_lab(image, label, self.args)
        image_trans = data_dict["image"]
        label_trans = data_dict["label"]
        if isinstance(image_trans, torch.Tensor):
            image_trans = image_trans.numpy()
        if isinstance(label_trans, torch.Tensor):
            label_trans = label_trans.numpy()

        # Only focus on vessels ...
        label_trans = np.where(label_trans == 2, 0, label_trans)
        label_trans = np.where(label_trans == 3, 2, label_trans)
        label_trans = np.where(label_trans == 4, 0, label_trans)
        label_trans = to_categorical(label_trans[0], 3)

        return image_trans, label_trans

    def __len__(self):
        return len(self.image_file)