# -*- coding: utf-8 -*-
from os.path import exists, join
from os import listdir
import numpy as np
import random
import fileinput
import os

"""
The purpose of this code is to generate ".txt" files in */TXT/ for training and testing
"""


def Get_file_list(file_dir):
    files = os.listdir(file_dir)
    # Sort files named with numbers (Like 1.nii.gz, 2.nii.gz)
    files.sort(key=lambda x: int(x.split(".")[0]))
    files_num = len(files)
    return files, files_num


def Generate_Txt(image_path, txt_name):
    f = open(txt_name, "w")
    files, files_num = Get_file_list(image_path)
    index_count = 0
    count = 0
    for file in files:
        index_count = index_count + 1
        if count == files_num - 1:
            f.write(image_path + str(file))
            break
        if index_count >= 0:
            f.write(image_path + str(file) + "\n")
            count = count + 1
    f.close()
    print("2 Finish Generate_Txt: ", txt_name)
