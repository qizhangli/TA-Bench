import os
import argparse
import logging

import torch

from dataset import NumpyImages
from utils import build_model, get_transforms


class Evaluation(object):
    def __init__(self, data_dir, mode="standard", victims=["tv_resnet50"],
                 batch_size=100, num_workers=4, log_dir=None):
        self.data_dir = data_dir
        self.dataloader = self._prep_data(data_dir, batch_size, num_workers)
        if mode == "standard":
            self.victims = ["tv_resnet50", "inception_v3", "vgg19_bn", "convnext_base", "vit_base_patch16_224", "deit3_base_patch16_224",
                 "beit_base_patch16_224", "swin_base_patch4_window7_224", "mixer_b16_224", "tf_efficientnetv2_m"]
        elif mode == "custom-custom":
            self._prep_victim = self._prep_victim_from_user
        elif mode == "custom-timm":
            self._prep_victim = self._prep_victim_from_timm
        else:
            raise RuntimeError("not support")
        self.log_dir = log_dir
        if self.log_dir:
            with open(self.log_dir, "w") as f:
                f.write("data_dir: {}\n".format(self.data_dir))
                
    def evaluate(self, ):
        print("Evaluation Start.")
        for victim in self.victims:
            model_name, model, transforms = self._prep_victim(victim)
            n_img, n_correct = 0, 0
            for img, label in self.dataloader:
                label, img = label.cuda(), img.cuda()
                with torch.no_grad():
                    pred = torch.argmax(model(transforms(img)), dim=1)
                n_correct += (label != pred).sum().item()
                n_img += len(label)
            suc_rate = 100 * n_correct / n_img
            log_str = "Victim Model: {}, Success Rate: {:.2f}%".format(model_name, suc_rate)
            if self.log_dir:
                with open(self.log_dir, "a") as f:
                    f.write(log_str+"\n")
            print(log_str)
        print("Evaluation Done.")
    
    def _prep_victim_from_timm(self, model_name):
        model, data_config = build_model(model_name)
        transforms = get_transforms(data_config, source=False)
        return model_name, model, transforms
        
    def _prep_victim_from_user(self, model_dict):
        return model_dict["model_name"], model_dict["model"], model_dict["preprocessing"]
    
    def _prep_data(self, data_dir, batch_size, num_workers):
        dataset = NumpyImages(data_dir)
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=batch_size, 
                                                 num_workers=num_workers, 
                                                 pin_memory=True)
        return dataloader

    
    
        