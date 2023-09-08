# -*- coding: utf-8 -*-
import os
import torch
import logging
import numpy as np
from os.path import join
import SimpleITK as sitk
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from S3_DSCNet import DSCNet
from S3_Dataloader import Dataloader
from S3_Loss import cross_loss

import warnings

warnings.filterwarnings("ignore")


# Use <AverageMeter> to calculate the mean in the process
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# One epoch in training process
def train_epoch(model, loader, optimizer, criterion, epoch, n_epochs):
    losses = AverageMeter()

    model.train()
    for batch_idx, (image, label) in enumerate(loader):
        if torch.cuda.is_available():
            image, label = image.cuda(), label.cuda()
        optimizer.zero_grad()
        model.zero_grad()

        output = model(image)
        loss = criterion(label, output)
        losses.update(loss.data, label.size(0))

        loss.backward()
        optimizer.step()

        res = "\t".join(
            [
                "Epoch: [%d/%d]" % (epoch + 1, n_epochs),
                "Iter: [%d/%d]" % (batch_idx + 1, len(loader)),
                "Lr: [%f]" % (optimizer.param_groups[0]["lr"]),
                "Loss %f" % (losses.avg),
            ]
        )
        print(res)
    return losses.avg


# Generate the log
def Get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# Train process
def Train_net(net, args):
    dice_mean, dice_m, dice_v, dice_a = 0, 0, 0, 0

    # Determine if trained parameters exist
    if not args.if_retrain and os.path.exists(
        os.path.join(args.Dir_Weights, args.model_name)
    ):
        net.load_state_dict(torch.load(os.path.join(args.Dir_Weights, args.model_name)))
        print(os.path.join(args.Dir_Weights, args.model_name))
    if torch.cuda.is_available():
        net = net.cuda()

    # Load dataset
    train_dataset = Dataloader(args)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.95))

    # It is possible to choose whether to use a dynamic learning rate,
    # which was not used in our original experiment, but you can choose to use
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=50)
    criterion = cross_loss()

    dt = datetime.today()
    log_name = (
        str(dt.date())
        + "_"
        + str(dt.time().hour)
        + ":"
        + str(dt.time().minute)
        + ":"
        + str(dt.time().second)
        + "_"
        + args.log_name
    )
    logger = Get_logger(args.Dir_Log + log_name)
    logger.info("start training!")

    # Main train process
    for epoch in range(args.start_train_epoch, args.n_epochs):
        loss = train_epoch(
            net, train_dataloader, optimizer, criterion, epoch, args.n_epochs
        )
        torch.save(net.state_dict(), os.path.join(args.Dir_Weights, args.model_name))
        # scheduler.step(loss)

        if epoch >= args.start_verify_epoch:
            net.load_state_dict(
                torch.load(os.path.join(args.Dir_Weights, args.model_name))
            )
            # The validation set is selected according to the task
            predict(net, args.Image_Te_txt, args.save_path, args)
            # Calculate the Dice
            dice_v, dice_a = Dice(args.Label_Te_txt, args.save_path)
            dice_v = np.mean(dice_v)
            dice_a = np.mean(dice_a)
            dice_mean = (dice_v + dice_a) / 2
            if dice_mean > dice_m:
                dice_m = dice_mean
                torch.save(
                    net.state_dict(),
                    os.path.join(args.Dir_Weights, args.model_name_max),
                )
        logger.info(
            "Epoch:[{}/{}]  lr={:.6f}  loss={:.5f}  dice_mean={:.4f} "
            "max_dice={:.4f}".format(
                epoch,
                args.n_epochs,
                optimizer.param_groups[0]["lr"],
                loss,
                dice_mean,
                dice_m,
            )
        )
    logger.info("finish training!")


def read_file_from_txt(txt_path):  # 从txt里读取数据
    files = []
    for line in open(txt_path, "r"):
        files.append(line.strip())
    return files


def reshape_img(image, z, y, x):
    out = np.zeros([z, y, x], dtype=np.float32)
    out[0 : image.shape[0], 0 : image.shape[1], 0 : image.shape[2]] = image[
        0 : image.shape[0], 0 : image.shape[1], 0 : image.shape[2]
    ]
    return out


# Predict process
def predict(model, image_dir, save_path, args):
    print("Predict test data")
    model.eval()
    file = read_file_from_txt(image_dir)
    file_num = len(file)

    for t in range(file_num):
        image_path = file[t]
        print(image_path)

        image = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image)
        image = image.astype(np.float32)

        name = image_path[image_path.rfind("/") + 1 :]
        mean, std = np.load(args.root_dir + args.Te_Meanstd_name)
        image = (image - mean) / std
        z, y, x = image.shape
        z_old, y_old, x_old = z, y, x

        if args.ROI_shape[0] > z:
            z = args.ROI_shape[0]
            image = reshape_img(image, z, y, x)
        if args.ROI_shape[1] > y:
            y = args.ROI_shape[1]
            image = reshape_img(image, z, y, x)
        if args.ROI_shape[2] > x:
            x = args.ROI_shape[2]
            image = reshape_img(image, z, y, x)

        predict = np.zeros([1, args.n_classes, z, y, x], dtype=np.float32)
        n_map = np.zeros([1, args.n_classes, z, y, x], dtype=np.float32)

        """
        Our prediction is carried out using sliding patches, 
        and for each patch a corresponding result is predicted, 
        and for the part where the patches overlap, 
        we use weight <map_kernel> balance, 
        and we agree that the closer to the center of the patch, the higher the weight
        """

        shape = args.ROI_shape
        a = np.zeros(shape=shape)
        a = np.where(a == 0)
        map_kernal = 1 / (
            (a[0] - shape[0] // 2) ** 4
            + (a[1] - shape[1] // 2) ** 4
            + (a[2] - shape[2] // 2) ** 4
            + 1
        )
        map_kernal = np.reshape(map_kernal, newshape=(1, 1,) + shape)

        # print(np.max(map_kernal))
        image = image[np.newaxis, np.newaxis, :, :, :]
        stride_x = shape[0] // 2
        stride_y = shape[1] // 2
        stride_z = shape[2] // 2
        for i in range(z // stride_x - 1):
            for j in range(y // stride_y - 1):
                for k in range(x // stride_z - 1):
                    image_i = image[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                              k * stride_z:k * stride_z + shape[2]]
                    image_i = torch.from_numpy(image_i)
                    if torch.cuda.is_available():
                        image_i = image_i.cuda()
                    output = model(image_i)
                    output = output.data.cpu().numpy()

                    predict[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                    k * stride_z:k * stride_z + shape[2]] += output * map_kernal

                    n_map[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                    k * stride_z:k * stride_z + shape[2]] += map_kernal

                image_i = image[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                          x - shape[2]:x]
                image_i = torch.from_numpy(image_i)
                if torch.cuda.is_available():
                    image_i = image_i.cuda()
                output = model(image_i)
                output = output.data.cpu().numpy()
                predict[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                x - shape[2]:x] += output * map_kernal

                n_map[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                x - shape[2]:x] += map_kernal

            for k in range(x // stride_z - 1):
                image_i = image[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y,
                          k * stride_z:k * stride_z + shape[2]]
                image_i = torch.from_numpy(image_i)
                if torch.cuda.is_available():
                    image_i = image_i.cuda()
                output = model(image_i)
                output = output.data.cpu().numpy()
                predict[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y,
                k * stride_z:k * stride_z + shape[2]] += output * map_kernal

                n_map[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y,
                k * stride_z:k * stride_z + shape[2]] += map_kernal

            image_i = image[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y, x - shape[2]:x]
            image_i = torch.from_numpy(image_i)
            if torch.cuda.is_available():
                image_i = image_i.cuda()
            output = model(image_i)
            output = output.data.cpu().numpy()

            predict[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y, x - shape[2]:x] += output * map_kernal
            n_map[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y, x - shape[2]:x] += map_kernal

        for j in range(y // stride_y - 1):
            for k in range((x - shape[2]) // stride_z):
                image_i = image[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
                          k * stride_z:k * stride_z + shape[2]]
                image_i = torch.from_numpy(image_i)
                if torch.cuda.is_available():
                    image_i = image_i.cuda()
                output = model(image_i)
                output = output.data.cpu().numpy()

                predict[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
                k * stride_z:k * stride_z + shape[2]] += output * map_kernal

                n_map[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
                k * stride_z:k * stride_z + shape[2]] += map_kernal

            image_i = image[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
                      x - shape[2]:x]
            image_i = torch.from_numpy(image_i)
            if torch.cuda.is_available():
                image_i = image_i.cuda()
            output = model(image_i)
            output = output.data.cpu().numpy()

            predict[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
            x - shape[2]:x] += output * map_kernal

            n_map[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
            x - shape[2]:x] += map_kernal

        for k in range(x // stride_z - 1):
            image_i = image[:, :, z - shape[0]:z, y - shape[1]:y,
                      k * stride_z:k * stride_z + shape[2]]
            image_i = torch.from_numpy(image_i)
            if torch.cuda.is_available():
                image_i = image_i.cuda()
            output = model(image_i)
            output = output.data.cpu().numpy()

            predict[:, :, z - shape[0]:z, y - shape[1]:y,
            k * stride_z:k * stride_z + shape[2]] += output * map_kernal

            n_map[:, :, z - shape[0]:z, y - shape[1]:y,
            k * stride_z:k * stride_z + shape[2]] += map_kernal

        image_i = image[:, :, z - shape[0]:z, y - shape[1]:y, x - shape[2]:x]
        image_i = torch.from_numpy(image_i)
        if torch.cuda.is_available():
            image_i = image_i.cuda()
        output = model(image_i)
        output = output.data.cpu().numpy()

        predict[:, :, z - shape[0]:z, y - shape[1]:y, x - shape[2]:x] += output * map_kernal
        n_map[:, :, z - shape[0]:z, y - shape[1]:y, x - shape[2]:x] += map_kernal

        predict = predict / n_map
        predict = np.argmax(predict[0], axis=0)
        predict = predict.astype(np.uint16)
        out = predict[0:z_old, 0:y_old, 0:x_old]
        out = sitk.GetImageFromArray(out)
        sitk.WriteImage(out, join(save_path, name))
    print("finish!")


def Dice(label_dir, pred_dir):
    # 获取image文件索引
    file = read_file_from_txt(label_dir)
    file_num = len(file)
    i = 0
    dice_vein = np.zeros(shape=(file_num), dtype=np.float32)
    dice_artery = np.zeros(shape=(file_num), dtype=np.float32)

    for t in range(file_num):
        image_path = file[t]
        name = image_path[image_path.rfind('/') + 1:]
        predict = sitk.ReadImage(join(pred_dir, name))
        predict = sitk.GetArrayFromImage(predict)

        groundtruth = sitk.ReadImage(image_path)
        groundtruth = sitk.GetArrayFromImage(groundtruth)

        groundtruth = np.where(groundtruth == 2, 0, groundtruth)
        groundtruth = np.where(groundtruth == 3, 2, groundtruth)
        groundtruth = np.where(groundtruth == 4, 0, groundtruth)

        predict_vein = np.where(predict == 1, 1, 0).flatten()
        predict_artery = np.where(predict == 2, 1, 0).flatten()
        groundtruth_vein = np.where(groundtruth == 1, 1, 0).flatten()
        groundtruth_artery = np.where(groundtruth == 2, 1, 0).flatten()

        tmp = predict_vein + groundtruth_vein
        a = np.sum(np.where(tmp == 2, 1, 0))
        b = np.sum(predict_vein)
        c = np.sum(groundtruth_vein)
        dice_vein[i] = (2 * a) / (b + c)

        tmp = predict_artery + groundtruth_artery
        a = np.sum(np.where(tmp == 2, 1, 0))
        b = np.sum(predict_artery)
        c = np.sum(groundtruth_artery)
        dice_artery[i] = (2 * a) / (b + c)
        print(name, dice_vein[i], dice_artery[i])
        i += 1

    return dice_vein, dice_artery


def Create_files(args):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(args.save_path_max):
        os.mkdir(args.save_path_max)


def Predict_Network(net, args):
    if torch.cuda.is_available():
        net = net.cuda()
    try:
        net.load_state_dict(
            torch.load(os.path.join(args.Dir_Weights, args.model_name_max))
        )
        print(os.path.join(args.Dir_Weights, args.model_name_max))
    except:
        print(
            "Warning 100: No parameters in weights_max, here use parameters in weights"
        )
        net.load_state_dict(torch.load(os.path.join(args.Dir_Weights, args.model_name)))
        print(os.path.join(args.Dir_Weights, args.model_name))
    predict(net, args.Image_Te_txt, args.save_path_max, args)
    dice = Dice(args.Label_Te_txt, args.save_path_max)
    dice_mean = np.mean(dice)
    print(dice_mean)


def Train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = DSCNet(
        n_channels=args.n_channels,
        n_classes=args.n_classes,
        kernel_size=args.kernel_size,
        extend_scope=args.extend_scope,
        if_offset=args.if_offset,
        device=device,
        number=args.n_basic_layer,
        dim=args.dim,
    )
    Create_files(args)
    if not args.if_onlytest:
        Train_net(net, args)
        Predict_Network(net, args)
    else:
        Predict_Network(net, args)



