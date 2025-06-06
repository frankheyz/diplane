import numpy as np
from skimage import io
from tifffile import imread
from tifffile import imsave
import torch
import os
from PIL import Image
from torchvision import transforms


def read_image(img_path, to_grayscale=True):
    # load image
    if img_path.endswith('.tif') and io.imread(img_path).shape.__len__() != 2:
        img = imread(img_path)
        if len(img.shape) == 3 and img.shape[0] >= 1:
            img = np.moveaxis(img, 0, -1)  # move the z-axis to the last dimension
        # convert numpy img to tensor
        img = torch.from_numpy(img)
        # convert 8-bits image to float
        img = img / 255
    else:
        img = Image.open(img_path)

    if to_grayscale and 'convert' in dir(img):
        img = img.convert('L')

    # # convert to tensor
    # if not isinstance(img, torch.Tensor):
    #     img = transforms.ToTensor()(img)

    return img


def save_tensor_as_img(input_tensor, save_path, save_name):

    if input_tensor.device.type != 'cpu':
        input_tensor = input_tensor.cpu()

    input_tensor = input_tensor.numpy() * 255
    input_tensor = input_tensor.astype('uint8')
    input_tensor = np.moveaxis(input_tensor, -1, 0)
    imsave(os.path.join(save_path, save_name), input_tensor)
