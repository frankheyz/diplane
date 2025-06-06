import copy
import os
import warnings

import cv2

from torch.utils.data import Dataset
import numpy as np
from math import pi
from scipy.io import loadmat
import PIL.Image
import torch
import torchio as tio
from torch import nn
from configs import configs
from utils_dir.training_image_io import read_image
from utils_dir.input_output_matching import get_x_y_no, input_output_matching
from utils_dir.multiscale_morph import multiscale_morph
from utils_dir.boundary_finding import boundary_finding
from skimage.transform import rescale
from scipy.ndimage import gaussian_filter, convolve
from torchvision import transforms
from scipy.ndimage import filters, measurements, interpolation
import random
import torchvision.transforms.functional as TF


class ZVisionDataset(Dataset):
    """
    Super-resolution dataset
    """
    base_sf = 1.0

    def __init__(self, configs=configs, perform_transform=None):
        """
        init function
        :param configs: dict
        :param transform:
        """
        self.configs = configs

        # get file dir
        self.img_dir = configs['image_path']
        self.img_names = os.listdir(self.img_dir)
        self.img_names = [f for f in self.img_names if f.endswith('.tif')]
        self.reference_img_dir = configs['reference_img_path']

        # calculate X tile and Y tile number
        self.x_no, self.y_no = get_x_y_no(self.img_dir)

        self.perform_transform = perform_transform

    def __len__(self):
        return len(self.img_names)

    def transform(self, image):
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=self.configs['crop_size']
        )

        cropped = TF.crop(image, i, j, h, w)

        return transforms.ToTensor()(cropped), i, j, h, w

    def __getitem__(self, idx):
        # read input
        img_path = os.path.join(
            self.img_dir, self.img_names[idx]
        )
        image = read_image(img_path)
        cv2_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # 1st focal map calculation
        focus_map = multiscale_morph(cv2_img) / 255
        focus_map = PIL.Image.fromarray(focus_map)
        focus_map = transforms.ToTensor()(focus_map)

        # image = transforms.ToTensor()(image)

        # read the second image as input if configured
        image_second = None
        focus_map_second = None
        if self.configs['plane_setting'] == 2:  # Diplane
            # process the second input
            image_second_path = self.get_second_input(
                self.img_dir, self.img_names[idx], self.configs['axial_distance']
            )
            image_second = read_image(image_second_path)
            cv2_img_second = cv2.imread(image_second_path, cv2.IMREAD_GRAYSCALE)

            # check if there is a dominant image, if yes, duplicate it
            img_no, dominant_idx = boundary_finding(
                img1_raw=cv2_img, img2_raw=cv2_img_second
            )
            if img_no == 1:  # means there is a dominant image
                if dominant_idx == 1:
                    cv2_img_second = cv2_img
                    image_second = image
                else:
                    cv2_img = cv2_img_second
                    image = image_second

            # convert it to tensor
            image_second = transforms.ToTensor()(image_second)

            # 2nd focal map calculation
            focus_map_second = multiscale_morph(cv2_img_second) / 255
            focus_map_second = PIL.Image.fromarray(focus_map_second)
            focus_map_second = transforms.ToTensor()(focus_map_second)
        elif self.configs['plane_setting'] == 1:  # Single plane
            # do nothing
            pass
        else:
            raise ValueError("Incorrect plane setting, please use 1 or 2")

        # if self.configs['shape_in'][1] == 4:
        #     image_second_path = self.get_second_input(
        #         self.img_dir, self.img_names[idx], self.configs['axial_distance']
        #     )
        #     image_second = read_image(image_second_path)
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
        #
        #     # convert it to tensor
        #     image_second = transforms.ToTensor()(image_second)
        #     focus_map_second = transforms.ToTensor()(focus_map_second)

        # read target
        target_path = input_output_matching(
            self.img_dir, self.reference_img_dir, self.y_no, idx
        )
        target = read_image(target_path)
        target = transforms.ToTensor()(target)

        # transform input image for augmentation
        if self.perform_transform is True:
            image_transformed, i, j, h, w = self.transform(image)  # note image has to be PIL or TIO subject
            focus_map_transformed = TF.crop(focus_map, i, j, h, w)
            target_transformed = TF.crop(target, i, j, h, w)

            # transform the second input when it exists
            if image_second is not None:
                image_second_transformed = TF.crop(image_second, i, j, h, w)
                focus_map_second_transformed = TF.crop(focus_map_second, i, j, h, w)


        else:
            image_transformed = transforms.ToTensor()(image)
            focus_map_transformed = transforms.ToTensor()(focus_map)
            target_transformed = transforms.ToTensor()(target)

            # transform the second input when it exists
            if image_second is not None:
                image_second_transformed = transforms.ToTensor()(image_second)
                focus_map_second_transformed = transforms.ToTensor()(focus_map_second)

        # extract data from subject
        if isinstance(image_transformed, tio.data.subject.Subject):
            image_transformed = image_transformed['sample'][tio.DATA]
            target_transformed = target_transformed['sample'][tio.DATA]

            if image_second is not None:
                image_second_transformed =image_second_transformed['sample'][tio.DATA]


        # Finally -  cat img with focus map
        if self.configs['plane_setting'] == 2:  # Diplane

            if self.configs['focal_map'] == 1:
                cat_img_focus_map = torch.cat(
                    (image_transformed, focus_map_transformed,
                     image_second_transformed, focus_map_second_transformed), 0
                )
            else:
                cat_img_focus_map = torch.cat(
                    (image_transformed,
                     image_second_transformed), 0
                )
        elif self.configs['plane_setting'] == 1:  # Single plane

            if self.configs['focal_map'] == 1:
                cat_img_focus_map = torch.cat((image_transformed, focus_map_transformed), 0)
            else:
                cat_img_focus_map = image_transformed
        else:
            raise ValueError("Incorrect plane_setting or focal_map setting.")

        if configs['model_name'] == 'SimVP':
            # add a dummy axis for time
            cat_img_focus_map = cat_img_focus_map.unsqueeze(0)
            target_transformed = target_transformed.unsqueeze(0)

        sample = {
            "img": cat_img_focus_map,
            # "focus_map": focus_map_transformed,
            "img_target": target_transformed
        }

        return sample

    @staticmethod
    def get_second_input(dir_path, img_name, axial_distance):
        """
        get image from the second camera
        :return: the path of the second input image
        """
        split_img_name = img_name.split('_')
        positive_file_name = copy.copy(split_img_name)
        negative_file_name = copy.copy(split_img_name)

        axial_pos = split_img_name[3]
        positive_second_pos = float(axial_pos) + axial_distance
        negative_second_pos = float(axial_pos) - axial_distance

        positive_file_name[3] = "{:.3f}".format(positive_second_pos)
        negative_file_name[3] = "{:.3f}".format(negative_second_pos)

        positive_file_name = "_".join(positive_file_name)
        negative_file_name = "_".join(negative_file_name)

        if os.path.exists(os.path.join(dir_path, positive_file_name)):
            return os.path.join(dir_path, positive_file_name)
        elif os.path.exists(os.path.join(dir_path, positive_file_name)) is False:
            # return os.path.join(dir_path, negative_file_name)
            # if the range exceed the limit, replicate the input
            warnings.warn("File exceed range. Replicate the input.")
            return os.path.join(dir_path, img_name)


def locate_smallest_axis(img):
    """
    find the index of the smallest axis (usually the z-axis is the smallest)
    :param img: a torch tensor
    :return: an index
    """
    dims = img.shape
    dim_list = list(dims)
    smallest_axis = dim_list.index(min(dim_list))

    return smallest_axis


def resize_tensor_along_dim(tenser_in, dim, weights, field_of_view):
    # To be able to act on each dim, we swap so that dim 0 is the wanted dim to resize
    tmp_tensor_in = torch.transpose(tenser_in, dim, 0)
    unchanged_dimensions_shape = list(tmp_tensor_in.shape)[1:]

    # We add singleton dimensions to the weight matrix so we can multiply it with the big tensor we get for
    # tmp_im[field_of_view.T], (bsxfun style)
    weights = np.reshape(weights.T, list(weights.T.shape) + (np.ndim(tenser_in) - 1) * [1])

    # This is a bit of a complicated multiplication: tmp_im[field_of_view.T] is a tensor of order image_dims+1.
    # for each pixel in the output-image it matches the positions the influence it from the input image (along 1 dim
    # only, this is why it only adds 1 dim to the shape). We then multiply, for each pixel, its set of positions with
    # the matching set of weights. we do this by this big tensor element-wise multiplication (MATLAB bsxfun style:
    # matching dims are multiplied element-wise while singletons mean that the matching dim is all multiplied by the
    # same number
    field_of_view = field_of_view.astype('int32')
    new_dims = field_of_view.T.shape[0]
    leading_dimensions_shape = [new_dims, field_of_view.T.shape[1]]
    new_tensor_shape = tuple(leading_dimensions_shape + unchanged_dimensions_shape)
    # new_tensor_shape = (new_dims, field_of_view.T.shape[1], tmp_tensor_in.shape[dim])
    if tmp_tensor_in.device.type == 'cpu':
        new_tensor = torch.zeros(new_tensor_shape)
    else:
        weights = torch.from_numpy(weights).float().to(tmp_tensor_in.device)
        new_tensor = torch.cuda.FloatTensor(*new_tensor_shape).fill_(0)

    if len(new_tensor.shape) == 3:
        for i in range(new_dims):
            new_tensor[i, :, :] = tmp_tensor_in[field_of_view.T[i]]
    elif len(new_tensor.shape) == 4:
        for i in range(new_dims):
            new_tensor[i, :, :, :] = tmp_tensor_in[field_of_view.T[i]]
    else:
        raise ValueError('Incorrect tensor dimensions. Please check the dimensions of the new tensor.')

    tmp_out_im = torch.sum(new_tensor * weights, dim=0)

    # Finally we swap back the axes to the original order
    return torch.transpose(tmp_out_im, dim, 0)


def resize_tensor(
        tensor_in,
        scale_factor=None,
        output_shape=None,
        kernel=None,
        antialiasing=True,
        kernel_shift_flag=False
        ):
    # convert it to tensor if it is an PIL image
    if isinstance(tensor_in, PIL.Image.Image):
        tensor_in = transforms.ToTensor()(tensor_in)
        tensor_in = torch.squeeze(tensor_in)

    if len(tensor_in.shape) == 3:
        three_d_image = True
    else:
        three_d_image = False
    # First standardize values and fill missing arguments (if needed) by deriving scale from output shape or vice versa
    scale_factor, output_shape = fix_scale_and_size(tuple(tensor_in.shape), output_shape, scale_factor, three_d_image)

    # For a given numeric kernel case, just do convolution and sub-sampling (downscaling only)
    if type(kernel) == np.ndarray and scale_factor[0] <= 1:

        return numeric_kernel(tensor_in, kernel, scale_factor, output_shape, kernel_shift_flag)

    # Choose interpolation method, each method has the matching kernel size
    method, kernel_width = {
        "cubic": (cubic, 4.0),
        "lanczos2": (lanczos2, 4.0),
        "lanczos3": (lanczos3, 6.0),
        "box": (box, 1.0),
        "linear": (linear, 2.0),
        None: (cubic, 4.0)  # set default interpolation method as cubic
    }.get(kernel)

    # Antialiasing is only used when downscaling
    antialiasing *= (scale_factor[0] < 1)

    # Sort indices of dimensions according to scale of each dimension. since we are going dim by dim this is efficient
    sorted_dims = np.argsort(np.array(scale_factor)).tolist()

    # Iterate over dimensions to calculate local weights for resizing and resize each time in one direction
    tensor_out = tensor_in.clone()
    for dim in sorted_dims:
        # No point doing calculations for scale-factor 1. nothing will happen anyway
        if scale_factor[dim] == 1.0:
            continue

        # for each coordinate (along 1 dim), calculate which coordinates in the input image affect its result and the
        # weights that multiply the values there to get its result.
        weights, field_of_view = contributions(tensor_in.shape[dim], output_shape[dim], scale_factor[dim],
                                               method, kernel_width, antialiasing)

        # Use the affecting position values and the set of weights to calculate the result of resizing along this 1 dim
        tensor_out = resize_tensor_along_dim(tensor_out, dim, weights, field_of_view)

    return tensor_out

def cubic(x):
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return ((1.5*absx3 - 2.5*absx2 + 1) * (absx <= 1) +
            (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * ((1 < absx) & (absx <= 2)))


def lanczos2(x):
    return (((np.sin(pi*x) * np.sin(pi*x/2) + np.finfo(np.float32).eps) /
             ((pi**2 * x**2 / 2) + np.finfo(np.float32).eps))
            * (abs(x) < 2))


def box(x):
    return ((-0.5 <= x) & (x < 0.5)) * 1.0


def lanczos3(x):
    return (((np.sin(pi*x) * np.sin(pi*x/3) + np.finfo(np.float32).eps) /
            ((pi**2 * x**2 / 3) + np.finfo(np.float32).eps))
            * (abs(x) < 3))


def linear(x):
    return (x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))


def kernel_shift(kernel, sf):
    # There are two reasons for shifting the kernel:
    # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
    #    the degradation process included shifting so we always assume center of mass is center of the kernel.
    # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
    #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
    #    top left corner of the first pixel. that is why different shift size needed between od and even size.
    # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
    # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The second ("+ 0.5 * ....") is for applying condition 2 from the comments above
    wanted_center_of_mass = np.array(kernel.shape) / 2 + 0.5 * (sf - (kernel.shape[0] % 2))

    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass

    # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
    # (biggest shift among dims + 1 for safety)
    kernel = np.pad(kernel, np.int(np.ceil(np.max(shift_vec))) + 1, 'constant')

    # Finally shift the kernel and return
    return interpolation.shift(kernel, shift_vec)


def numeric_kernel(im, kernel, scale_factor, output_shape, kernel_shift_flag):
    # See kernel_shift function to understand what this is
    if kernel_shift_flag:
        kernel = kernel_shift(kernel, scale_factor)

    # First run a correlation (convolution with flipped kernel)
    # out_im = np.zeros_like(im)
    # if len(scale_factor) == 2:
    #     for channel in range(np.ndim(im)):
    #         out_im[:, :, channel] = filters.correlate(im[:, :, channel], kernel)
    # else:
    in_device = im.device
    if im.device.type == 'cuda':  # when making output, the im is in cuda
        im = im.cpu()

    out_im = filters.correlate(im, kernel)
    out_im_tensor = torch.from_numpy(out_im).to(in_device)

    # Then subsample and return
    if len(scale_factor) == 2:
        return out_im_tensor[np.round(np.linspace(0, im.shape[0] - 1 / scale_factor[0], output_shape[0])).astype(int)[:, None],
                      np.round(np.linspace(0, im.shape[1] - 1 / scale_factor[1], output_shape[1])).astype(int)]
    else:
        return out_im_tensor[
                # 1st dim
                np.round(np.linspace(0, im.shape[0] - 1 / scale_factor[0], output_shape[0])).astype(int)[:, None, None],
                # 2nd dim
                np.round(np.linspace(0, im.shape[1] - 1 / scale_factor[1], output_shape[1])).astype(int)[None, :, None],
                # 3rd dim
                np.round(np.linspace(0, im.shape[2] - 1 / scale_factor[2], output_shape[2])).astype(int)
               ]


def fspecial3(shape=(3,3,3), sigma=(1,1,1)):
    """
    matlab style 3D gaussian mask
    """
    i, j, k = [(s-1.)/2. for s in shape]
    y, x, z = np.ogrid[-i:i+1, -j:j+1, -k:k+1]

    xsig2 = sigma[1] ** 2
    ysig2 = sigma[0] ** 2
    zsig2 = sigma[2] ** 2

    arg = -(x**2 / (2 * xsig2) + y**2 / (2 * ysig2) + z**2 / (2 * zsig2))
    h = np.exp(arg)
    h[h<np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()

    if sumh != 0:
        h /= sumh
    return h


def fix_scale_and_size(input_shape, output_shape, scale_factor, three_d_image=False):
    # First fixing the scale-factor (if given) to be standardized the function expects (a list of scale factors in the
    # same size as the number of input dimensions)
    if scale_factor is not None:
        # By default, if scale-factor is a scalar we assume 2d resizing and duplicate it.
        if np.isscalar(scale_factor) and three_d_image is False:
            scale_factor = [scale_factor, scale_factor]
        elif np.isscalar(scale_factor) and three_d_image is True:
            scale_factor = [scale_factor, scale_factor, scale_factor]

        # We extend the size of scale-factor list to the size of the input by assigning 1 to all the unspecified scales
        scale_factor = list(scale_factor)
        scale_factor.extend([1] * (len(input_shape) - len(scale_factor)))

    # Fixing output-shape (if given): extending it to the size of the input-shape, by assigning the original input-size
    # to all the unspecified dimensions
    if output_shape is not None:
        output_shape = list(np.uint(np.array(output_shape))) + list(input_shape[len(output_shape):])

    # Dealing with the case of non-give scale-factor, calculating according to output-shape. note that this is
    # sub-optimal, because there can be different scales to the same output-shape.
    if scale_factor is None:
        scale_factor = 1.0 * np.array(output_shape) / np.array(input_shape)

    # Dealing with missing output-shape. calculating according to scale-factor
    if output_shape is None:
        output_shape = np.uint(np.ceil(np.array(input_shape) * np.array(scale_factor)))

    return scale_factor, output_shape


def contributions(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    # This function calculates a set of 'filters' and a set of field_of_view that will later on be applied
    # such that each position from the field_of_view will be multiplied with a matching filter from the
    # 'weights' based on the interpolation method and the distance of the sub-pixel location from the pixel centers
    # around it. This is only done for one dimension of the image.

    # When anti-aliasing is activated (default and only for downscaling) the receptive field is stretched to size of
    # 1/sf. this means filtering is more 'low-pass filter'.
    fixed_kernel = (lambda arg: scale * kernel(scale * arg)) if antialiasing else kernel
    kernel_width *= 1.0 / scale if antialiasing else 1.0

    # These are the coordinates of the output image
    out_coordinates = np.arange(1, out_length+1)

    # These are the matching positions of the output-coordinates on the input image coordinates.
    # Best explained by example: say we have 4 horizontal pixels for HR and we downscale by SF=2 and get 2 pixels:
    # [1,2,3,4] -> [1,2]. Remember each pixel number is the middle of the pixel.
    # The scaling is done between the distances and not pixel numbers (the right boundary of pixel 4 is transformed to
    # the right boundary of pixel 2. pixel 1 in the small image matches the boundary between pixels 1 and 2 in the big
    # one and not to pixel 2. This means the position is not just multiplication of the old pos by scale-factor).
    # So if we measure distance from the left border, middle of pixel 1 is at distance d=0.5, border between 1 and 2 is
    # at d=1, and so on (d = p - 0.5).  we calculate (d_new = d_old / sf) which means:
    # (p_new-0.5 = (p_old-0.5) / sf)     ->          p_new = p_old/sf + 0.5 * (1-1/sf)
    match_coordinates = 1.0 * out_coordinates / scale + 0.5 * (1 - 1.0 / scale)

    # This is the left boundary to start multiplying the filter from, it depends on the size of the filter
    left_boundary = np.floor(match_coordinates - kernel_width / 2)

    # Kernel width needs to be enlarged because when covering has sub-pixel borders, it must 'see' the pixel centers
    # of the pixels it only covered a part from. So we add one pixel at each side to consider (weights can zeroize them)
    expanded_kernel_width = np.ceil(kernel_width) + 2

    # Determine a set of field_of_view for each each output position, these are the pixels in the input image
    # that the pixel in the output image 'sees'. We get a matrix whos horizontal dim is the output pixels (big) and the
    # vertical dim is the pixels it 'sees' (kernel_size + 2)
    field_of_view = np.squeeze(np.uint(np.expand_dims(left_boundary, axis=1) + np.arange(expanded_kernel_width) - 1))

    # Assign weight to each pixel in the field of view. A matrix whos horizontal dim is the output pixels and the
    # vertical dim is a list of weights matching to the pixel in the field of view (that are specified in
    # 'field_of_view')
    weights = fixed_kernel(1.0 * np.expand_dims(match_coordinates, axis=1) - field_of_view - 1)

    # Normalize weights to sum up to 1. be careful from dividing by 0
    sum_weights = np.sum(weights, axis=1)
    sum_weights[sum_weights == 0] = 1.0
    weights = 1.0 * weights / np.expand_dims(sum_weights, axis=1)

    # We use this mirror structure as a trick for reflection padding at the boundaries
    mirror = np.uint(np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))))
    field_of_view = mirror[np.mod(field_of_view, mirror.shape[0])]

    # Get rid of  weights and pixel positions that are of zero weight
    non_zero_out_pixels = np.nonzero(np.any(weights, axis=0))
    weights = np.squeeze(weights[:, non_zero_out_pixels])
    field_of_view = np.squeeze(field_of_view[:, non_zero_out_pixels])

    # Final products are the relative positions and the matching weights, both are output_size X fixed_kernel_size
    return weights, field_of_view


def back_project_tensor(y_sr, y_lr, down_kernel, up_kernel, sf=None):
    """
    Use back projection technique to reduce super resolution error
    :param y_sr:
    :param y_lr:
    :param down_kernel:
    :param up_kernel:
    :param sf:
    :return:
    """

    y_sr_low_res_projection = resize_tensor(y_sr,
                                            scale_factor=1.0 / sf,
                                            output_shape=y_lr.shape,
                                            kernel=down_kernel)
    y_sr += resize_tensor(y_lr - y_sr_low_res_projection,
                          scale_factor=sf,
                          output_shape=y_sr.shape,
                          kernel=up_kernel)

    # if not isinstance(down_kernel, str) or not isinstance(up_kernel, str):
    #     raise TypeError("Unimplemented resizing methods.")
    #
    # # add batch, channel dimensions
    # y_sr.unsqueeze_(0).unsqueeze_(0)
    # y_lr.unsqueeze_(0).unsqueeze_(0)
    #
    # y_sr_low_res_projection = interpolate(y_sr, scale_factor=1.0/sf[0])
    # y_sr += interpolate(y_lr - y_sr_low_res_projection, scale_factor=sf[0])
    #
    # y_sr = y_sr.squeeze()

    # return torch.clamp(y_sr, 0, 1)
    return torch.clamp_min(y_sr, 0)


def valid_image_region(input_img, configs):
    """
    crop image to avoid edge effect.
    :param input_img:
    :param configs:
    :return:
    """
    cut_size = configs['kernel_dilation'] * (configs['kernel_size'] - 1) / 2

    if input_img.shape.__len__() == 2:
        return input_img[int(cut_size): int(-1-cut_size), int(cut_size): int(-1-cut_size)]
    elif input_img.shape.__len__() == 3:
        return input_img[
               int(cut_size): int(-1-cut_size),
               int(cut_size): int(-1-cut_size),
               int(cut_size): int(-1-cut_size)
               ]
    else:
        raise ValueError("Incorrect input image size.")


class RotationTransform:
    """
    Rotate the input image by one of the given angles
    """
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class RandomCrop3D:
    def __init__(self, crop_size: tuple):
        self.crop_size = crop_size

    def __call__(self, input_tensor):
        sampler = tio.data.UniformSampler(self.crop_size)
        patch = sampler(input_tensor, 1)

        return list(patch)[0]


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
