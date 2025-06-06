import glob
import json
import os
from shutil import copy as copy_file

import cv2
import torch
import PIL.Image
from torchvision import transforms
from torch import nn
from modules import ConvSC, Inception
from utils_dir.boundary_finding import boundary_finding
from utils_dir.training_image_io import read_image
from utils_dir.multiscale_morph import multiscale_morph

from torchvision.utils import save_image
from tifffile import imsave
import numpy as np
from math import log10, sqrt
import warnings

from tabulate import tabulate

from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from utils_dir.training_dataset import locate_smallest_axis
from utils_dir.training_dataset import valid_image_region
from utils_dir.training_dataset import ZVisionDataset


def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse: return list(reversed(strides[:N]))
    else: return strides[:N]


class Encoder(nn.Module):
    def __init__(self,C_in, C_hid, N_S):
        super(Encoder,self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )
    
    def forward(self,x):# B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1,len(self.enc)):
            latent = self.enc[i](latent)
        return latent,enc1


# update output channel for uvblur dataset
class Decoder(nn.Module):
    def __init__(self,C_hid, C_out, N_S):
        super(Decoder,self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)
    
    def forward(self, hid, enc1=None):
        for i in range(0,len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y


class Mid_Xnet(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker = [3,5,7,11], groups=8):
        super(Mid_Xnet, self).__init__()

        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_in, incep_ker= incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y


class SimVP(nn.Module):
    def __init__(self, configs, incep_ker=[3,5,7,11], groups=8):
        super(SimVP, self).__init__()
        self.configs = configs
        shape_in = self.configs['shape_in']
        shape_out = self.configs['shape_out']
        hid_S = self.configs['hid_S']
        hid_T = self.configs['hid_T']
        N_S = self.configs['N_S']
        N_T = self.configs['N_T']
        T, C, H, W = shape_in
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Mid_Xnet(T*hid_S, hid_T, N_T, incep_ker, groups)
        if shape_out is None:
            self.c_out = C
        else:
            self.c_out = shape_out[1]

        self.dec = Decoder(hid_S, self.c_out, N_S)

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape # batch, time, channel, height, width
        x = x_raw.view(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B*T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, self.c_out, H, W)
        return Y

    def output(self):
        # read image
        input_img = read_image(self.configs['test_img_path'], self.configs['to_grayscale'])
        cv2_img = cv2.imread(self.configs['test_img_path'], cv2.IMREAD_GRAYSCALE)


        # do not crop the image when testing
        self.configs["crop_size"] = tuple(self.configs['shape_in'][2:])

        if self.configs['plane_setting'] == 2:  # Diplane
            # read the second image
            get_second_input = ZVisionDataset.get_second_input
            img_dir = self.configs['test_img_path'].split('/')[0:-1]
            img_dir = "/".join(img_dir)
            img_name = self.configs['test_img_path'].split('/')[-1]
            image_second_path = get_second_input(
                img_dir, img_name, self.configs['axial_distance']
            )
            image_second = read_image(image_second_path)
            # calculate focus map
            cv2_img_second = cv2.imread(image_second_path, cv2.IMREAD_GRAYSCALE)

            # TODO check if there is a dominant image, if yes, duplicate it
            img_no, dominant_idx = boundary_finding(
                img1_raw=cv2_img, img2_raw=cv2_img_second
            )
            print("--------------------------")
            print(img_name)
            print(image_second_path.split('/')[-1])
            if img_no == 1:  # means there is a dominant image
                if dominant_idx == 1:
                    cv2_img_second = cv2_img
                    image_second = input_img
                    print("Dominant: input 1 ", img_name)
                else:
                    # replace input 1 with input 2
                    cv2_img = cv2_img_second
                    input_img = image_second
                    print("Dominant: input 2 ", image_second_path)
                    pass
            print("--------------------------")


            if self.configs['focal_map'] == 1:
                # calculate focal map of input 1
                focus_map = multiscale_morph(cv2_img) / 255
                focus_map = PIL.Image.fromarray(focus_map)
                focus_map = transforms.ToTensor()(focus_map)
                # calculate focal map of input 2
                focus_map_second = multiscale_morph(cv2_img_second) / 255
                focus_map_second = PIL.Image.fromarray(focus_map_second)
                focus_map_second = transforms.ToTensor()(focus_map_second)

                # cat input
                input_img = torch.cat(
                    (transforms.ToTensor()(input_img), focus_map,
                     transforms.ToTensor()(image_second), focus_map_second), 0
                )
            else:
                # cat input
                input_img = torch.cat(
                    (transforms.ToTensor()(input_img),
                     transforms.ToTensor()(image_second)), 0
                )
        elif self.configs['plane_setting'] == 1:  # Single plane

            if self.configs['focal_map'] == 1:
                focus_map = multiscale_morph(cv2_img) / 255
                focus_map = PIL.Image.fromarray(focus_map)
                focus_map = transforms.ToTensor()(focus_map)
                # cat input
                input_img = torch.cat(
                    (transforms.ToTensor()(input_img), focus_map), 0
                )
            else:
                pass

        # if self.configs['shape_in'][1] == 1:
        #     pass
        # elif self.configs['shape_in'][1] == 2:
        #     input_img = torch.cat((transforms.ToTensor()(input_img), focus_map), 0)
        # elif self.configs['shape_in'][1] == 4:
        #     get_second_input = ZVisionDataset.get_second_input
        #     img_dir = self.configs['test_img_path'].split('/')[0:-1]
        #     img_dir = "/".join(img_dir)
        #     img_name = self.configs['test_img_path'].split('/')[-1]
        #     image_second_path = get_second_input(
        #         img_dir, img_name, self.configs['axial_distance']
        #     )
        #     image_second = read_image(image_second_path)
        #     # calculate focus map
        #     cv2_img_second = cv2.imread(image_second_path, cv2.IMREAD_GRAYSCALE)
        #
        #     # TODO check if there is a dominant image, if yes, duplicate it
        #     img_no, dominant_idx = boundary_finding(
        #         img1_raw=cv2_img, img2_raw=cv2_img_second
        #     )
        #     if img_no == 1:  # means there is a dominant image
        #         if dominant_idx == 1:
        #             cv2_img_second = cv2_img
        #         else:
        #             pass
        #
        #     focus_map_second = multiscale_morph(cv2_img_second) / 255
        #     focus_map_second = PIL.Image.fromarray(focus_map_second)
        #     focus_map_second = transforms.ToTensor()(focus_map_second)
        #     # cat input
        #     input_img = torch.cat(
        #         (transforms.ToTensor()(input_img), focus_map,
        #          transforms.ToTensor()(image_second), focus_map_second), 0
        #     )
        # else:
        #     raise IOError("Incorrect input shape for model inference.")

        self.output_tensor(input_img)

        # save
        self.save_outputs()

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
                       + '.' + self.configs['test_img_path'].split('.')[-1]
            self.output_img_path = os.path.join(out_path, out_name)
            if out_name.endswith('jpg') or out_name.endswith('png'):
                out_img = self.final_output #/ torch.max(self.final_output)
                save_image(out_img, self.output_img_path)
            elif out_name.endswith('tif'):
                # save as tif.
                out_img = self.final_output.cpu().numpy()
                # clip
                out_img = np.clip(out_img, 0, 1)
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

        if self.configs['copy_code']:
            local_dir = os.path.dirname(__file__)
            for py_file in glob.glob(local_dir + '/*.py'):
                copy_file(py_file, self.configs['save_path'])

    def output_tensor(self, input_img):
        self.eval()
        if isinstance(input_img, PIL.Image.Image):
            input_img = transforms.ToTensor()(input_img)

        input_img = input_img.unsqueeze(0)
        input_img = input_img.unsqueeze(0)
        input_img = input_img.to('cuda')
        with torch.no_grad():
            network_out = self.__call__(input_img)
            # undo processing(rotation, flip, etc.)
            network_out = network_out.squeeze()
            # add a singleton dimension to make sure flipping is the same
            network_out = network_out.unsqueeze(0)
            self.final_output = network_out

        return self.final_output

    def evaluate_error(self):
        # mse, ssim etc.
        # format output
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

