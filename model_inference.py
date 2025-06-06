# -*- coding: utf-8 -*-
import os
import json
import time
import torch
from models import ZVision
from models import ZVisionMini
from models import count_parameters
from configs import configs
from simplevp_model import SimVP


def infer_image(conf, image_dir, model_path, save_path, model='SimVP'):
    print('model', model)
    # set gpu
    dev = torch.device(
        "cuda:0" if torch.cuda.is_available() and configs['use_gpu']
        else torch.device("cpu")
    )
    conf['save_path'] = os.path.join(save_path, 'test_results'
                                     # time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
                                     )
    out = None

    # load the image and the model
    conf['test_img_path'] = image_dir

    if model == 'SimVP':
        model = SimVP(configs=conf)
    else:
        model = ZVision(configs=conf)

    print("model parameters: ", count_parameters(model))
    model.load_state_dict(torch.load(model_path))

    # model to device
    model.to(dev)
    if dev.type == 'cuda':
        model.dev = dev

    model.eval()
    # inference
    out = model.output()

    return out


if __name__ == "__main__":
    from natsort import natsorted
    test_image_dir = '/home/heyz/data/uv_refocusing/z210-240_8bits'
    model_path = './results/20240131_20_01_46_mouse_brain_pressed_dual_cam_two_channel_e50_ad10_v3/checkpoint/SimVP'
    save_path = './results/image_inferred/z210-240_8bits_network_output'
    model = 'SimVP'

    images = os.listdir(test_image_dir)
    images = natsorted(images)
    # load configs saved
    with open(os.path.join(model_path.rstrip('checkpoint/SimVP'), 'output_configs/configs.json')) as f:
        configs = json.load(f)

    # update crop size
    configs["crop_size"] = tuple(configs['shape_in'][2:])

    for img in images:
        img_path = os.path.join(test_image_dir, img)
        infer_image(
            conf=configs,
            image_dir=img_path,
            model_path=model_path,
            model=model,
            save_path=save_path
        )
