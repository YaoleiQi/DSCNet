# -*- coding: utf-8 -*-
import os
import argparse

from S1_Pre_Getmeanstd import Getmeanstd
from S2_Pre_Generate_Txt import Generate_Txt
from S3_Train_Process import Train

"""
This code contains all the "Parameters" for the entire project -- <DSCNet>
Code Introduction: (The easiest way to run a code!)
    !!! You just need to change lines with "# todo" to get straight to run
    !!! Our code is encapsulated, but it also provides some test interfaces for debugging
    !!! If you want to change the dataset, you can change "KIPA" to other task name
    
KIPA22 [1-4] challenge (including simulataneous segmentation of arteries and veins) is used as 
a public 3D dataset to further validate our method
Challenge: https://kipa22.grand-challenge.org/

[1] He, Y. et. al. 2021. Meta grayscale adaptive network for 3D integrated renal structures segmentation. 
Medical image analysis 71, 102055.
[2] He, Y. et. al. 2020. Dense biased networks with deep priori anatomy and hard region adaptation: 
Semisupervised learning for fine renal artery segmentation. Medical Image Analysis 63, 101722.
[3] Shao, P. et. al. 2011. Laparoscopic partial nephrectomy with segmental renal artery clamping: 
technique and clinical outcomes. European urology 59, 849–855.
[4] Shao, P. et. al. 2012. Precise segmental renal artery clamping under the guidance of dual-source computed 
tomography angiography during laparoscopic partial nephrectomy. European urology 62, 1001–1008.
"""


def Create_files(args):
    print("0 Start all process ...")
    if not os.path.exists(args.Dir_Txt):
        os.makedirs(args.Dir_Txt)
    if not os.path.exists(args.Dir_Log):
        os.makedirs(args.Dir_Log)
    if not os.path.exists(args.Dir_Save):
        os.makedirs(args.Dir_Save)
    if not os.path.exists(args.Dir_Weights):
        os.makedirs(args.Dir_Weights)


def Process(args):
    # step 0: Prepare all files in this projects
    Create_files(args)

    # Step 1: Prepare image and calculate the "mean" and "std" for normalization
    Getmeanstd(args, args.Tr_Image_dir, args.Tr_Meanstd_name)
    Getmeanstd(args, args.Te_Image_dir, args.Te_Meanstd_name)

    # Step 2: Prepare ".txt" files for training and testing data
    Generate_Txt(args.Tr_Image_dir, args.Image_Tr_txt)
    Generate_Txt(args.Te_Image_dir, args.Image_Te_txt)
    Generate_Txt(args.Tr_Label_dir, args.Label_Tr_txt)
    Generate_Txt(args.Te_Label_dir, args.Label_Te_txt)

    # Step 3: Train the "Network"
    Train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # "root_dir" refers to the address of the outermost code, and "***" needs to be replaced
    root_dir = "***/DSCNet_3D_opensource/"                              # todo
    data_dir = '***/Data/KIPA/'                                         # todo
    parser.add_argument(
        "--root_dir", default=root_dir, help="the address of the outermost code"
    )

    # information about the image and label
    parser.add_argument(
        "--Tr_Image_dir",
        default=data_dir + "train/image/",
        help="the address of the train image",
    )
    parser.add_argument(
        "--Te_Image_dir",
        default=data_dir + "open/image/",
        help="the address of the test image",
    )
    parser.add_argument(
        "--Tr_Label_dir",
        default=data_dir + "train/label/",
        help="the address of the train label",
    )
    parser.add_argument(
        "--Te_Label_dir",
        default=data_dir + "open/label/",
        help="the address of the test label",
    )
    parser.add_argument(
        "--Tr_Meanstd_name",
        default="KIPA_Tr_Meanstd.npy",
        help="Train image Mean and std for normalization",
    )
    parser.add_argument(
        "--Te_Meanstd_name",
        default="KIPA_Te_Meanstd.npy",
        help="Test image Mean and std for normalization",
    )

    # files that are needed to be used to store contents
    parser.add_argument(
        "--Dir_Txt", default=root_dir + "Txt/Txt_KIPA/", help="Txt path"
    )
    parser.add_argument("--Dir_Log", default=root_dir + "Log/KIPA/", help="Log path")
    parser.add_argument(
        "--Dir_Save", default=root_dir + "Results/KIPA/", help="Save path"
    )
    parser.add_argument(
        "--Dir_Weights", default=root_dir + "Weights/KIPA/", help="Weights path"
    )

    # Folders, dataset, etc.
    parser.add_argument(
        "--Image_Tr_txt",
        default=root_dir + "Txt/Txt_KIPA/Image_Tr.txt",
        help="train image txt path",
    )
    parser.add_argument(
        "--Image_Te_txt",
        default=root_dir + "Txt/Txt_KIPA/Image_Te.txt",
        help="test image txt path",
    )
    parser.add_argument(
        "--Label_Tr_txt",
        default=root_dir + "Txt/Txt_KIPA/Label_Tr.txt",
        help="train label txt path",
    )
    parser.add_argument(
        "--Label_Te_txt",
        default=root_dir + "Txt/Txt_KIPA/Label_Te.txt",
        help="test label txt path",
    )

    # Detailed path for saving results
    """
    Breif description:
        Due to the small proportion of the thin tubular structure, 
        the results of the model may bring huge fluctuations. 
        In order to reduce the influence of uncertain factors on the model analysis, 
        we save the <best> results on the validation dataset in the <max> folder, 
        and apply the same standard to all comparative methods to ensure fairness!!
    """
    parser.add_argument(
        "--save_path", default=root_dir + "Results/KIPA/DSCNet/", help="Save dir"
    )
    parser.add_argument(
        "--save_path_max",
        default=root_dir + "Results/KIPA/DSCNet_max/",
        help="Save max dir",
    )
    parser.add_argument("--model_name", default="DSCNet_KIPA", help="Weights name")
    parser.add_argument(
        "--model_name_max", default="DSCNet_KIPA_max", help="Max Weights name"
    )
    parser.add_argument("--log_name", default="DSCNet_KIPA.log", help="Log name")

    # Network options
    parser.add_argument("--n_channels", default=1, type=int, help="input channels")
    parser.add_argument("--n_classes", default=3, type=int, help="output channels")
    parser.add_argument(
        "--kernel_size", default=9, type=int, help="kernel size"
    )  # 9 refers to 1*9/9*1 for DSConv
    parser.add_argument(
        "--extend_scope", default=1.0, type=float, help="extend scope"
    )  # This parameter is not used
    parser.add_argument(
        "--if_offset", default=True, type=bool, help="if offset"
    )  # Whether to use the deformation or not
    parser.add_argument(
        "--n_basic_layer", default=16, type=int, help="basic layer numbers"
    )
    parser.add_argument("--dim", default=8, type=int, help="dim numbers")

    # Training options
    parser.add_argument("--GPU_id", default="0", help="GPU ID")                             # todo
    """
    Reference: --ROI_shape: (128, 96, 96)  3090's memory occupancy is about 16653 MiB
    """
    parser.add_argument("--ROI_shape", default=(128, 96, 96), type=int, help="roi size")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--lr", default=1e-4, type=int, help="learning rate")
    parser.add_argument(
        "--start_train_epoch", default=0, type=int, help="Start training epoch"
    )
    parser.add_argument(
        "--start_verify_epoch", default=200, type=int, help="Start verifying epoch"
    )
    parser.add_argument("--n_epochs", default=400, type=int, help="Epoch Num")
    parser.add_argument("--if_retrain", default=True, type=bool, help="If Retrain")
    parser.add_argument("--if_onlytest", default=False, type=bool, help="If Only Test")

    args, unknown = parser.parse_known_args()
    Process(args)