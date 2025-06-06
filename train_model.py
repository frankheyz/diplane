# -*- coding: utf-8 -*-
import os
import sys
import argparse
import torch
from torch import nn

from models import get_model
from configs import configs as conf

from utils_dir.training_utils import Logger
from utils_dir.training_dataset import ZVisionDataset
from utils_dir.training_utils import fit
from utils_dir.training_utils import get_data
from utils_dir.training_utils import get_transform


def serial_training(serial_count):
    def decorator(func):
        def wrapper(*args, **kwargs):
            trained_model = None
            # train for different scales
            for i in range(serial_count):
                trained_model = func(*args, **kwargs)
                new_training_img = trained_model.output_img_path
                # update config
                if i < serial_count - 1:
                    kwargs['configs']['image_path'] = new_training_img

            return trained_model
        return wrapper

    return decorator


@serial_training(serial_count=conf['serial_training'])
def train_model(configs=conf, checkpoint_dir=None):
    # set random seed
    torch.manual_seed(configs['manual_seed_num'])
    # set gpu
    dev = torch.device(
        "cuda:0" if torch.cuda.is_available() and configs['use_gpu']
        else torch.device("cpu")
    )

    # todo use deterministic transform
    # compose transforms
    composed_transform = get_transform(configs=configs)
    # todo optimize loading of the input for data parallelism
    # define train data set and data loader
    train_ds = ZVisionDataset(
        configs=configs, perform_transform=True
    )
    train_size = int(configs['train_split']*len(train_ds))
    test_size = len(train_ds) - train_size
    train_ds, valid_ds = torch.utils.data.random_split(train_ds, [train_size, test_size])

    # define train data set and data loader
    # valid_ds = ZVisionDataset(
    #     configs=configs, perform_transform=True
    # )

    train_dl, valid_dl = get_data(
        train_ds, valid_ds, configs=configs
    )

    # get model and optimizer
    model, opt = get_model(configs=configs)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint")
        )
        model.load_state_dict(model_state)
        opt.load_state_dict(optimizer_state)

    # todo Data parallel
    if torch.cuda.device_count() > 2:
        print("Using", torch.cuda.device_count(), "GPUs.")
        model = nn.DataParallel(model)

    print('Input image: ', configs['image_path'])

    model.to(device=dev)

    # fit the model
    trained_model = fit(
        configs=configs,
        model=model,
        loss_func=nn.L1Loss() if configs['loss_func'] == 'l1' else nn.MSELoss(),
        opt=opt,
        train_dl=train_dl,
        valid_dl=valid_dl,
        device=dev,
    )

    trained_model.output()

    if 'tune' in configs:
        return
    else:
        return trained_model


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Train z-vision model.")
    parser.add_argument("-m", "--model", type=str, help="choose model", default='up')

    parser.add_argument("-i", "--image_path", type=str, help="Input image path.", default=None)
    parser.add_argument("-r", "--reference_img_path", type=str, help="Reference image path.", default=None)
    parser.add_argument("-e", "--max_epoches", type=int, help="Epoches", default=100)
    parser.add_argument("-ad", "--axial_distance", type=int, help="axial distance", default=None)
    parser.add_argument("-ps", "--plane_setting", type=int, help="Plane setting: 1,2", default=2)
    parser.add_argument("-f", "--focal_map", type=int, help="Use focal map: 1 for true 0 for false", default=1)
    parser.add_argument("-n", "--notes", type=str, help="Add notes.", default=None)

    args = parser.parse_args()

    input_config = conf
    input_config['model'] = 'up' if args.model.lower() == 'up' else 'Original model'
    input_config['image_path'] = args.image_path if args.image_path is not None else input_config['image_path']
    input_config['reference_img_path'] = args.reference_img_path \
        if args.reference_img_path is not None else input_config['reference_img_path']
    input_config['axial_distance'] = args.axial_distance if args.axial_distance is not None else input_config['axial_distance']
    input_config['max_epochs'] = args.max_epoches if args.max_epoches is not None else input_config['max_epochs']
    input_config['focal_map'] = args.focal_map if args.focal_map is not None else input_config['focal_map']
    input_config['plane_setting'] = args.plane_setting if args.plane_setting is not None else input_config['plane_setting']

    if input_config['plane_setting'] == 2:  # Diplane
        if input_config['focal_map'] == 1:
            input_config['shape_in'][1] = 4
        else:
            input_config['shape_in'][1] = 2
    elif input_config['plane_setting'] == 1:  # Single plane
        if input_config['focal_map'] == 1:
            input_config['shape_in'][1] = 2
        else:
            input_config['shape_in'][1] = 1


    if args.notes:
        print(args.notes)
        input_config['save_path'] = input_config['save_path'] + '_' + args.notes + '/'

    print("model:", args.model.lower())

    # logger
    path = input_config['save_path']
    sys.stdout = Logger(path)
    m = train_model(configs=input_config)

    m.evaluate_error()

    # todo add ssim to objective function
    # todo percentile normalization
    # todo demonstrate the advantage of image specific network

    # todo fix output cubic bug when kernel provided
    # todo fix memory issue
