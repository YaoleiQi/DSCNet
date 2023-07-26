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


class Dataloader(data.Dataset):
    def __init__(self, args):
        super(Dataloader, self).__init__()
        self.image_file = read_file_from_txt(args.Image_Tr_txt)
        self.label_file = read_file_from_txt(args.Label_Tr_txt)
        self.shape = (args.ROI_shape, args.ROI_shape)
        self.args = args

    def __getitem__(self, index):
        image_path = self.image_file[index]
        label_path = self.label_file[index]

        # Read images and labels
        image = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image)
        label = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(label)

        y, x = image.shape
        image = image.astype(dtype=np.float32)
        label = label.astype(dtype=np.float32)

        # Normalization
        mean, std = np.load(self.args.root_dir + self.args.Tr_Meanstd_name)
        image = (image - mean) / std
        label = label / (np.max(label))

        # Random crop, (center_y, center_x) refers the left-up coordinate of the Random_Crop_Block
        center_y = np.random.randint(0, y - self.shape[0] + 1, 1, dtype=np.int16)[0]
        center_x = np.random.randint(0, x - self.shape[1] + 1, 1, dtype=np.int16)[0]
        image = image[
            center_y : self.shape[0] + center_y, center_x : self.shape[1] + center_x
        ]
        label = label[
            center_y : self.shape[0] + center_y, center_x : self.shape[1] + center_x
        ]

        image = image[np.newaxis, :, :]
        label = label[np.newaxis, :, :]

        # Data Augmentation
        data_dict = transform_img_lab(image, label, self.args)
        image_trans = data_dict["image"]
        label_trans = data_dict["label"]
        if isinstance(image_trans, torch.Tensor):
            image_trans = image_trans.numpy()
        if isinstance(label_trans, torch.Tensor):
            label_trans = label_trans.numpy()

        return image_trans, label_trans

    def __len__(self):
        return len(self.image_file)
