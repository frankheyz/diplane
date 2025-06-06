# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

import glob
from shutil import copy as copy_file
import json
import os

import cv2
from tifffile import imsave

import PIL.Image
import torch
import numpy as np
from scipy.io import loadmat
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

from configs import configs as conf
from utils_dir.input_output_matching import input_output_matching
from utils_dir.multiscale_morph import multiscale_morph

from utils_dir.training_image_io import read_image
from utils_dir.training_dataset import resize_tensor
from utils_dir.training_dataset import locate_smallest_axis
from utils_dir.training_dataset import back_project_tensor
from utils_dir.training_dataset import valid_image_region
from utils_dir.training_dataset import Interpolate
from utils_dir.training_utils import count_parameters

from simplevp_model import SimVP

import math
from math import log10, sqrt
import warnings

from tabulate import tabulate

from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim


class ZVision(nn.Module):
    base_sf = 1.0
    scale_factor = None
    scale_factor_idx = 0
    final_output = None

    def __init__(self, configs=conf):
        super(ZVision, self).__init__()
        self.configs = configs
        self.dev = torch.device(
            "cuda:0" if torch.cuda.is_available() and configs['use_gpu']
            else torch.device("cpu")
        )
        self.original_img_tensor = None
        self.output_img_path = None
        self.scale_factor = np.array(configs['scale_factor']) / np.array(self.base_sf)
        self.upscale_method = self.configs['upscale_method']

        # select 2D or 3D kernel
        self.kernel_selected = self.kernel_selector()
        # update padding
        pad_size = (self.configs['kernel_dilation'] * (self.configs['kernel_size'] - 1)) / 2
        self.configs['padding'] = (int(pad_size),) * 2 if self.configs['crop_size'].__len__() == 2 \
            else (int(pad_size), ) * 3
        self.conv_first = self.kernel_selected(
            in_channels=configs['input_channel_num'],
            out_channels=configs['kernel_channel_num'],
            kernel_size=configs['kernel_size'],
            stride=configs['kernel_stride'],
            dilation=configs['kernel_dilation'],
            padding=configs['padding'],
            padding_mode=configs['padding_mode']
        )
        self.conv_last = self.kernel_selected(
            in_channels=configs['kernel_channel_num'],
            out_channels=configs['output_channel_num'],
            kernel_size=configs['kernel_size'],
            stride=configs['kernel_stride'],
            dilation=configs['kernel_dilation'],
            padding=configs['padding'],
            padding_mode=configs['padding_mode']
        )

        # create layers
        layers = list()

        # the 1st layer
        layers.append(self.conv_first)

        # use a for loop to create hidden layers
        for i in range(1, self.configs['kernel_depth'] - 1):
            layers.append(
                self.kernel_selected(
                    in_channels=configs['kernel_channel_num'],
                    out_channels=configs['kernel_channel_num'],
                    kernel_size=configs['kernel_size'],
                    stride=configs['kernel_stride'],
                    dilation=configs['kernel_dilation'],
                    padding=configs['padding'],
                    padding_mode=configs['padding_mode'],
                )
            )

        # the last layers
        layers.append(self.conv_last)
        self.layers = nn.ModuleList(layers)

    # todo consider to remove it
    def kernel_selector(self):
        # determine if it is 2D or 3D using crop size
        if len(self.configs['crop_size']) == 2:
            return nn.Conv2d
        elif len(self.configs['crop_size']) == 3:
            return nn.Conv3d
        else:
            raise ValueError('Incorrect crop size. Please input a list of 2 or 3 elements.')

    def forward(self, xb):
        # interpolate xb to high resolution
        xb_instance_high_res_list = []
        batch_size = xb.size()[0]
        for i in range(batch_size):
            if self.configs['crop_size'].__len__() == 2:
                xb_instance = torch.squeeze(xb, 1)[i, :]  # todo fix here
            else:  # 3d case
                xb_instance = torch.squeeze(xb, 1)[i, :, :, :]

            xb_instance_high_res = resize_tensor(
                xb_instance,  # keep x,y dimensions only
                scale_factor=self.scale_factor,
                kernel=self.upscale_method
            )
            xb_instance_high_res_list.append(xb_instance_high_res)

        # turn xb_stack to tensor
        xb_instance_high_res_tensor = torch.stack(xb_instance_high_res_list)

        # TODO check which activation function is better
        # add channel dimensions (not x,y,z, batch)
        xb_high_res = xb_instance_high_res_tensor.unsqueeze(1).float()
        xb_mid = F.relu(self.layers[0](xb_high_res))

        for layer in range(1, self.configs['kernel_depth'] - 1):
            xb_mid = F.relu(self.layers[layer](xb_mid))

        xb_last = self.layers[-1](xb_mid)
        # output the last layer with residue
        xb_output = xb_last + self.configs['residual_learning'] * xb_high_res

        return torch.clamp(xb_output, 0, 1)
        # return torch.clamp_min(xb_output, 0)

    def output(self):
        # read image
        input_img = read_image(self.configs['test_img_path'], self.configs['to_grayscale'])
        if self.configs['input_channel_num'] == 2:
            cv2_img = cv2.imread(self.configs['test_img_path'], cv2.IMREAD_GRAYSCALE)
            focus_map = multiscale_morph(cv2_img) / 255
            focus_map = PIL.Image.fromarray(focus_map)
            focus_map = transforms.ToTensor()(focus_map)
            input_img = torch.cat((transforms.ToTensor()(input_img), focus_map), 0)

        self.output_tensor(input_img)

        # save
        self.save_outputs()

        return self.final_output

    def output_tensor(self, input_img: torch.Tensor):
        # load image
        swap_z = False
        z_index = 0
        # convert PIL image to tensor
        if isinstance(input_img, PIL.Image.Image):
            input_img_tensor = transforms.ToTensor()(input_img)
            input_img_tensor = input_img_tensor.unsqueeze(0)  # add the 'batch size' dimension
        elif isinstance(input_img, torch.Tensor):
            input_img_tensor = input_img
            # swap z-axis to the first dimension so that flip and rotations are perform in the x-y plane
            z_index = locate_smallest_axis(input_img_tensor)
            input_img_tensor = input_img_tensor.moveaxis(z_index, 0)
            input_img_tensor = input_img_tensor.unsqueeze(0)  # add the 'batch size' dimension
            swap_z = True
        else:
            raise ValueError("Incorrect input image format. Only PIL or torch.Tensor is allowed.")

        if self.dev:
            if self.dev.type == 'cuda':
                input_img_tensor = input_img_tensor.to('cuda')

            # run forward propagation
        self.eval()
        with torch.no_grad():
            if swap_z:
                # undo swapping, move z to the last axis for the model inference
                processed_input = torch.moveaxis(input_img_tensor, 1, -1)

            # output dimensions (1, 1, x, y) or (1, 1, x, y, z) [batch size, channel, l, w, h]
            if isinstance(self, ZVisionMini) and self.configs['crop_size'].__len__() == 3:
                # todo: fix here
                # this compensate line 144
                processed_input = input_img_tensor.unsqueeze(0)
            network_out = self.__call__(input_img_tensor)
            # undo processing(rotation, flip, etc.)
            network_out = network_out.squeeze()
            # add a singleton dimension to make sure flipping is the same
            network_out = network_out.unsqueeze(0)
            self.final_output = network_out

        return self.final_output

    def save_outputs(self):
        # save output img
        if self.configs['save_output_img'] is True:
            out_path = os.path.join(
                self.configs['save_path'],
                self.configs['output_img_dir']
            )
            os.makedirs(out_path, exist_ok=True)
            img_name = self.configs["test_img_path"].split('/')[-1]
            out_name = img_name[:-4] \
                       + ''.join('X%.2f' % s for s in self.configs['scale_factor']) \
                       + '.' + self.configs['test_img_path'].split('.')[-1]
            self.output_img_path = os.path.join(out_path, out_name)
            if out_name.endswith('jpg') or out_name.endswith('png'):
                out_img = self.final_output #/ torch.max(self.final_output)
                save_image(out_img, self.output_img_path)
            elif out_name.endswith('tif'):
                # save as tif.
                out_img = self.final_output.cpu().numpy()
                if out_img.max() > 1:
                    out_img = out_img / out_img.max()
                out_img = out_img * 255
                out_img = out_img.astype('uint8')
                imsave(self.output_img_path, out_img)
            else:
                raise TypeError("Invalid output image format.")

        if self.configs['save_configs'] is True:
            out_path = os.path.join(
                self.configs['save_path'],
                self.configs['output_configs_dir']
            )
            os.makedirs(out_path, exist_ok=True)
            # save(copy) config for reproducibility
            with open(out_path + "configs.json", 'w') as f:
                json.dump(self.configs, f, indent=4)

        if self.configs['use_provided_kernel'] and self.configs['save_kernel']:
            copy_file(self.configs["kernel_path"], self.configs['save_path'])

        if self.configs['copy_code']:
            local_dir = os.path.dirname(__file__)
            for py_file in glob.glob(local_dir + '/*.py'):
                copy_file(py_file, self.configs['save_path'])

    def half_split_output(self):
        """
        spit large image into half and output
        :return:
        """
        img = read_image(self.configs['image_path'], self.configs['to_grayscale'])

    def evaluate_error(self):
        # mse, ssim etc.
        # format output
        interp_factor = self.configs['serial_training'] * self.configs['scale_factor'][0]
        final_output_np = self.final_output.detach().cpu().numpy()
        # load reference image
        ref_path = self.configs['test_ref_img_path']
        ref_img = read_image(ref_path, self.configs['to_grayscale'])
        # ref_img = Image.open(ref_path).convert('L')
        ref_img = np.asarray(ref_img).astype(final_output_np.dtype)
        if locate_smallest_axis(ref_img) != locate_smallest_axis(final_output_np):
            # move the z axis to the last
            final_output_np = np.moveaxis(
                final_output_np, locate_smallest_axis(final_output_np), -1
            )

        ref_img_normalized = ref_img/np.max(ref_img)

        final_output_np = final_output_np.squeeze()
        if ref_img_normalized.shape != final_output_np.shape:
            warnings.warn(
                message='The output image shape does not match the reference. No evaluation was performed.'
            )

            return

        sr_mse = mean_squared_error(
            valid_image_region(ref_img_normalized, self.configs),
            valid_image_region(final_output_np, self.configs)
        )
        sr_ssim = ssim(
            valid_image_region(ref_img_normalized, self.configs),
            valid_image_region(final_output_np, self.configs)
        )
        sr_psnr = 20 * log10(1/sqrt(sr_mse))


        print(
            tabulate(
                [
                    ["MSE", "{:.6f}".format(sr_mse)],
                    ["SSIM", "{:.6f}".format(sr_ssim)],
                    ["PSNR", "{:.6f}".format(sr_psnr)],
                ],
                headers=['Errors', 'SRx2'],
                tablefmt='grid'
            )
        )

class ZVisionMini(ZVision):
    def __init__(self, configs):
        ZVision.__init__(self, configs=configs)
        in_channels = self.configs['input_channel_num']
        expansion_channels = self.configs['kernel_channel_num']
        shrinking = self.configs['shrinking']
        kernel_depth = self.configs['kernel_depth']
        first_kernel_size = self.configs['first_kernel_size']
        mid_kernel_size = self.configs['mid_kernel_size']
        last_kernel_size = self.configs['last_kernel_size']

        self.kernel_selected = self.kernel_selector()

        self.conv_first = nn.Sequential(
            self.kernel_selected(
                in_channels,
                expansion_channels,
                kernel_size=first_kernel_size,
                padding=first_kernel_size//2
            ),
            nn.PReLU(expansion_channels)
        )
        del self.layers
        self.layers = [
            self.kernel_selected(
                in_channels=expansion_channels,
                out_channels=shrinking,
                kernel_size=mid_kernel_size,
                padding=mid_kernel_size // 2,
                groups=1
            ),
            nn.PReLU(shrinking)
        ]
        for _ in range(kernel_depth):
            self.layers.extend(
                [
                    self.kernel_selected(
                        in_channels=shrinking,
                        out_channels=shrinking,
                        kernel_size=mid_kernel_size,
                        padding=mid_kernel_size//2,
                        groups=self.configs['groups']
                    ),
                    nn.PReLU(shrinking)
                ]
            )
        self.layers.extend(
            [
                self.kernel_selected(
                    in_channels=shrinking,
                    out_channels=expansion_channels,
                    kernel_size=3,
                    padding=3 // 2,
                    groups=1
                ),
                nn.PReLU(expansion_channels)
            ]
        )

        # self.conv_second_last = self.kernel_selected(
        #     in_channels=out_channels,
        #     out_channels=scale_factor ** self.configs['scale_factor'].__len__(),
        #     kernel_size=3,
        #     padding=3 // 2
        # )

        # todo consider to remove
        # self.conv_second_last = Interpolate(
        #     scale_factor=scale_factor,
        #     mode='nearest'
        # )
        #
        # self.layers.extend([self.conv_second_last])

        self.layers = nn.Sequential(*self.layers)

        self.conv_last = self.kernel_selected(
            in_channels=expansion_channels,
            out_channels=1,
            kernel_size=last_kernel_size,
            padding=last_kernel_size // 2
        )

        self._initialize_weights()

    # todo add conv3d init.?
    def _initialize_weights(self):
        for m in self.conv_first:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight[0][0].numel())))
                nn.init.zeros_(m.bias)
        for m in self.layers:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight[0][0].numel())))
                nn.init.zeros_(m.bias)
        if hasattr(self.conv_last, 'weight'):
            nn.init.normal_(self.conv_last.weight, mean=0.0, std=0.001)
        if hasattr(self.conv_last, 'bias'):
            nn.init.zeros_(self.conv_last.bias)

    def forward(self, x):
        x = x.float()
        x = self.conv_first(x)
        x = self.layers(x)
        x = self.conv_last(x)
        # todo residual learning
        return torch.clamp_min(x, 0)

    def transpose_kernel_selector(self):
        # determine if it is 2D or 3D using crop size
        if len(self.configs['crop_size']) == 2:
            return nn.ConvTranspose2d
        elif len(self.configs['crop_size']) == 3:
            return nn.ConvTranspose3d
        else:
            raise ValueError('Incorrect crop size. Please input a list of 2 or 3 elements.')


def get_model(configs):
    if configs['model'] == 'up':
        # model = ZVisionUp(configs=configs)
        # model = ZVisionMini(configs=configs)
        model = SimVP(configs=configs)
        print('{model_name} model'.format(model_name=model._get_name()))
        configs['model_name'] = model._get_name()
    else:
        model = ZVision(configs=configs)
        print('{model_name} model'.format(model_name=model._get_name()))

    print(
        count_parameters(model), "trainable parameters."
    )

    return model, optim.Adam(model.parameters(), lr=configs['learning_rate'])
