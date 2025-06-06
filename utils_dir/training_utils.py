import os
import sys
import time
import copy
import torch
import numpy as np
from ray import tune
from configs import configs as conf
import torchio as tio
from torch.utils.data import DataLoader
from torchvision import transforms

from utils_dir.training_image_io import read_image
from utils_dir.training_dataset import RotationTransform
from utils_dir.training_dataset import RandomCrop3D

from matplotlib import pyplot as plt
from torch.optim import lr_scheduler

from tqdm import tqdm


class Logger:
    def __init__(self, path, file_name='log.txt'):
        self.console = sys.stdout
        os.makedirs(path, exist_ok=True)
        self.file = open(
            os.path.join(path, file_name), 'a'
        )

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()


def get_data(train_ds, valid_ds, configs=conf):
    bs = configs['batch_size']
    num_work = configs['num_workers']

    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_work),
        DataLoader(valid_ds, batch_size=bs, num_workers=num_work),
    )


def get_transform(configs):
    # 2D case
    if len(configs['crop_size']) == 2:
        # # calculate image mean and std.
        # img = read_image(configs['image_path'])
        # img_tensor = transforms.ToTensor()(img)
        # img_tensor_mean = torch.mean(img_tensor).item()
        # img_tensor_std = torch.std(img_tensor).item()
        # add rotations
        rotation = RotationTransform(angles=configs['rotation_angles'])
        # compose transforms
        # todo 3d transform for crop etc.
        composed_transform = transforms.Compose([
            transforms.RandomCrop(configs['crop_size']),
            transforms.RandomHorizontalFlip(p=configs['horizontal_flip_probability']),
            transforms.RandomVerticalFlip(p=configs['vertical_flip_probability']),
            rotation,
            transforms.ToTensor(),
        ])

        # if configs['normalization']:
        #     composed_transform.transforms.append(transforms.Normalize(mean=img_tensor_mean, std=img_tensor_std))

    elif len(configs['crop_size']) == 3:  # 3D case
        normalization = tio.ZNormalization()
        (crop_x, crop_y, crop_z) = configs['crop_size']
        random_crop_3d = RandomCrop3D(crop_size=(crop_x, crop_y, crop_z))
        flips = tio.RandomFlip(axes=['LR', 'AP', 'IS'])
        # rotate about the z-axis
        rotations_dict = {
            tio.RandomAffine(scales=(1, 1, 1, 1, 1, 1), degrees=(0, 0, 0, 0, 90, 90)): 1 / 3,
            tio.RandomAffine(scales=(1, 1, 1, 1, 1, 1), degrees=(0, 0, 0, 0, 90 * 2, 90 * 2)): 1 / 3,
            tio.RandomAffine(scales=(1, 1, 1, 1, 1, 1), degrees=(0, 0, 0, 0, 90 * 3, 90 * 3)): 1 / 3,
        }
        rotation = tio.OneOf(rotations_dict)
        transforms_list = [random_crop_3d, flips, rotation]
        if configs['normalization']:
            transforms_list.insert(0, normalization)

        composed_transform = tio.Compose(transforms=transforms_list)

    else:
        raise Exception('Crop size invalid, please input 2D or 3D array.')

    return composed_transform


def fit(configs, model, loss_func, opt, train_dl, valid_dl, device=torch.device("cpu")):
    """
        Fit the network
        :param configs: training config
        :param model:
        :param loss_func:
        :param opt: optimizer
        :param train_dl: train data loader
        :param valid_dl: valid data loader
        :param device: cpu or gpu
        :param tuning: if it is called by ray tuning
        :return:
    """
    if device == torch.device("cpu"):
        print("**** Start training on CPU. ****")
    else:
        print("**** Start training on GPU. ****")

    start_time = time.time()
    loss_values = []
    min_loss = 1
    best_model = None
    best_epoch = 0
    save_path = os.path.join(configs['save_path'], configs['checkpoint_dir'])
    os.makedirs(save_path, exist_ok=True)

    rate_scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=opt,
        mode='min',
        factor=configs['adaptive_lr_factor'],
        min_lr=configs['min_lr'],
        verbose=True
    )

    for epoch in range(configs['max_epochs']):
        model.train()
        train_pbar = tqdm(train_dl)

        for _, sample in enumerate(train_pbar):
            xb = sample['img'].to(device)
            xb = xb.float()
            yb = sample['img_target'].to(device)  # todo updata here
            yb = yb.float()
            train_loss, train_num = loss_batch(model, loss_func, xb, yb, opt)
            train_pbar.set_description('train loss: {:.4f}'.format(train_loss))
            # print(train_loss)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(
                    model, loss_func, sample['img'].to(device), sample['img_target'].to(device)  # todo update here
                )
                    for _, sample in enumerate(valid_dl)]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        if val_loss < min_loss:
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            min_loss = val_loss

        loss_values.append(val_loss)

        if configs['adaptive_lr']:
            rate_scheduler.step(val_loss)

        if epoch % configs['show_loss'] == 0:
            print("epoch: {epoch}/{epochs}  validation loss: {loss:.8f}".format(
                epoch=epoch+1, epochs=configs['max_epochs'], loss=val_loss)
            )

        if epoch != 0 and epoch % configs['time_lapsed'] == 0:
            time_lapsed = time.time() - start_time
            print(
                "{epoch} epoch passed after {time_lapsed:.2f}".format(epoch=epoch, time_lapsed=time_lapsed)
            )

        if epoch != 0 and epoch % configs['checkpoint'] == 0:
            # save the model state dict
            torch.save(best_model.state_dict(), save_path + configs['model_name'])

        # report to tune
        if 'tune' in configs:
            tune.report(loss=val_loss)

    torch.save(best_model.state_dict(), save_path + configs['model_name'])

    plt.plot(loss_values)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Losses')
    loss_path = os.path.join(configs['save_path'],configs['output_img_dir'])
    loss_file = os.path.join(loss_path, 'loss_fig')
    os.makedirs(loss_path, exist_ok=True)
    plt.savefig(loss_file)
    # plt.show()

    print("Best epoch: ", best_epoch + 1)

    return best_model


def loss_batch(model, loss_func, xb, yb, opt=None):
    """
    Calculate the loss from a batch of samples
    :param model: input torch model
    :param loss_func:
    :param xb: input sample
    :param yb: target
    :param opt: optimizer
    :return: loss, sample size
    """
    # calculate loss
    loss = loss_func(model(xb), yb)  # model(xb) is the model output, yb is the target
    # from pytorch_msssim import ms_ssim
    # ms_ssim_val = ms_ssim(model(xb), yb, data_range=1)
    # loss = (1 - ms_ssim_val)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), len(xb)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

