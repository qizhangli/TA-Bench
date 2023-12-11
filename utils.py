import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as tvF
import torchvision.transforms as T
from PIL import Image

from dataset import SelectedImagenet
from timm.data.constants import *
import timm
from easydict import EasyDict
import yaml
import copy

def get_transforms(data_config, source):
    transforms = timm.data.transforms_factory.create_transform(
                        input_size = data_config['input_size'],
                        interpolation = data_config['interpolation'],
                        mean=(0,0,0),
                        std=(1,1,1),
                        crop_pct=data_config['crop_pct'] if not source else 1.,
                        tf_preprocessing=False,
                    )
    if not source:
        transforms.transforms = transforms.transforms[:-2]
    return transforms

def build_dataset(args, data_config, source=True):
    img_transform = get_transforms(data_config, source)
    dataset = SelectedImagenet(imagenet_val_dir=args.data_dir,
                               selected_images_csv=args.data_info_dir,
                               transform=img_transform)
    data_loader = torch.utils.data.DataLoader(dataset, 
                                              batch_size=args.batch_size, 
                                              shuffle=False, 
                                              pin_memory = True, 
                                              num_workers=args.num_workers)
    return data_loader
    
def build_model(model_name):
    model = timm.create_model(model_name, pretrained=True)
    data_config = model.pretrained_cfg
    model = nn.Sequential(T.Normalize(data_config["mean"], 
                                      data_config["std"]), 
                          model)
    model = nn.DataParallel(model)
    model.eval()
    model.cuda()
    return model, data_config

def build_generative_model(model_name, model_dir, method):
    data_config = {'input_size':(3, 224, 224),
                   'interpolation':'bilinear',
                   'crop_pct':1}
    if method in ["CDA", "GAPF", "BIA", "BIADA", "BIARN"]:
        from models.generators import GeneratorResnet
        netG = GeneratorResnet()
        state_dict = torch.load(model_dir)
        if method == "GAPF":
            state_dict = state_dict["model_state_dict"]
        netG.load_state_dict(state_dict)
        netG = nn.DataParallel(netG)
        netG.cuda()
        netG.eval()
        return netG, data_config
    elif method in ["CGSP"]:
        from models.condgenerators import ConGeneratorResnet
        netG = ConGeneratorResnet(nz=16, layer=1)
        state_dict = torch.load(model_dir)
        netG.load_state_dict(state_dict)
        netG = nn.DataParallel(netG)
        netG.cuda()
        netG.eval()
        return netG, data_config
    elif method in ["TTP"]:
        from models.generators import GeneratorResnet
        netG = GeneratorResnet()
        state_dict = torch.load(model_dir[0])
        netG.load_state_dict(state_dict)
        netG = nn.DataParallel(netG)
        netG.cuda()
        netG.eval()
        netG_2 = GeneratorResnet()
        state_dict = torch.load(model_dir[1])
        netG_2.load_state_dict(state_dict)
        netG_2 = nn.DataParallel(netG_2)
        netG_2.cuda()
        netG_2.eval()
        return (netG, netG_2), data_config


    