# -*- coding: utf-8 -*-
import time
import copy
from torch import nn

"""
   Configuration file for the network
"""
configs = {
    # data loading configs
    "image_path": "/home/heyz/data/diplane/pressed_top_n_8bits_training",
    "reference_img_path": "/home/heyz/data/diplane/pressed_top_n_8bits_merged",
    "test_img_path": "/home/heyz/data/diplane/pressed_top_n_8bits_test/Mosaic_11_2_200.000_250ms_A_1.tif",
    "test_ref_img_path": "/home/heyz/data/diplane/usaf_train_output/tile_0.tif",
    "data_format": 'jpg',
    "axial_distance": 10,
    "to_grayscale": True,
    "batch_size": 2,
    "num_workers": 0,
    "train_split": 0.8,

    # data preprocessing configs
    "plane_setting": 2,
    "focal_map": 1,
    "shape_in": [1, 1, 1024, 1024],
    "shape_out": [1, 1, 1024, 1024],
    "manual_seed_num": 1,
    "crop_size": (1024, 1024),
    "noise_std": 0.0,
    "rotation_angles": [90, 180, 270],
    "horizontal_flip_probability": 0.5,
    "vertical_flip_probability": 0.5,
    "output_flip": True,
    "back_projection_iters": [5],
    "normalization": False,

    # training hyper-parameters
    "model": 'old',
    "use_gpu": True,
    "serial_training": 1,
    "learning_rate": 0.00015,
    "adaptive_lr": False,
    "min_lr": 9e-6,
    "adaptive_lr_factor": 0.5,
    "loss_func": 'l2',
    "hid_S": 32,
    "hid_T": 32,
    "N_S": 4,
    "N_T": 8,
    "max_epochs": 1500,
    "min_epochs": 128,
    "show_loss": 1,
    # "input_channel_num": 2,
    "output_channel_num": 1,
    "kernel_depth": 8,
    "kernel_size": 3,
    "kernel_channel_num": 64,
    "kernel_stride": (1, 1),
    "kernel_dilation": 1,
    "groups": 1,
    "padding": (1, 1),  # padding size should be dilation x (kernel_size - 1) / 2 to achieve same convolution
    "padding_mode": 'reflect',
    'background_threshold': 0.1,
    'background_percentage': 0.25,
    "time_lapsed": 100,
    "residual_learning": True,
    'interp_method': 'cubic',

    # ZVisionMini parameters
    'shrinking': 12,
    'mid_layers': 4,
    'first_kernel_size': 5,
    'mid_kernel_size': 3,
    'last_kernel_size': 9,

    # save configs
    "configs_file_path": __file__,
    "checkpoint": 500,
    "save_path": '/home/heyz/code/diplane/results/' + time.strftime("%Y%m%d_%H_%M_%S", time.localtime()),
    "model_dir": 'model/',
    "checkpoint_dir": 'checkpoint/',
    "save_output_img": True,
    "output_img_dir": 'output_image/',
    "model_name": "model.pt",
    "save_configs": True,
    "save_kernel": True,
    "copy_code": True,
    "output_configs_dir": 'output_configs/',
    "max_pixels": 1.13e7
}
