import numpy as np
import torch

from .helper import update_and_clip

__all__ = ["CDA", "GAPF", "BIA", "BIADA", "BIARN", "CGSP", "TTP"]

class Generative(object):
    def __init__(self, args, **kwargs):
        self.model = kwargs["source_model"]
    
    def __call__(self, args, ori_img, label, model, verbose=True):
        with torch.no_grad():
            adv_img = self.model(ori_img)
        adv_img = update_and_clip(ori_img, adv_img, torch.zeros(1).to(adv_img.device), 
                                  args.epsilon, 0, args.constraint)
        return adv_img
    
class CDA(Generative):
    def __init__(self, args, **kwargs):
        Generative.__init__(self, args, **kwargs)

class GAPF(Generative):
    def __init__(self, args, **kwargs):
        Generative.__init__(self, args, **kwargs)

class BIA(Generative):
    def __init__(self, args, **kwargs):
        Generative.__init__(self, args, **kwargs)

class BIADA(Generative):
    def __init__(self, args, **kwargs):
        Generative.__init__(self, args, **kwargs)

class BIARN(Generative):
    def __init__(self, args, **kwargs):
        Generative.__init__(self, args, **kwargs)

class CGSP(Generative):
    def __init__(self, args, **kwargs):
        Generative.__init__(self, args, **kwargs)
        self.class_ids = np.array([150, 507, 62, 843, 426, 590, 715, 952])

    def __call__(self, args, ori_img, label, model, verbose=True):
        target_tensor = torch.LongTensor(ori_img.size(0))
        while True:
            rand_target_index = np.random.choice(self.class_ids)
            if (label == rand_target_index).sum() == 0:
                break
        target_tensor.fill_(rand_target_index)
        target_one_hot = torch.zeros(ori_img.size(0), 1000).scatter_(1, target_tensor.unsqueeze(1), 1).cuda()
        with torch.no_grad():
            adv_img = ori_img+self.model(ori_img, target_one_hot, eps=16/255)
        adv_img = update_and_clip(ori_img, adv_img, torch.zeros(1).to(adv_img.device), 
                                  args.epsilon, 0, args.constraint)
        return adv_img
    
class TTP(Generative):
    def __init__(self, args, **kwargs):
        Generative.__init__(self, args, **kwargs)
        self.ttp_target = args.ttp_target
        
    def __call__(self, args, ori_img, label, model, verbose=True):
        if (label==self.ttp_target[0]).sum() > 0:
            adv_img = torch.zeros_like(ori_img).to(ori_img.device)
            ori_img_1 = ori_img[label!=self.ttp_target[0]].clone()
            ori_img_2 = ori_img[label==self.ttp_target[0]].clone()
            with torch.no_grad():
                adv_img_1 = self.model[0](ori_img_1)
            adv_img_1 = update_and_clip(ori_img_1, adv_img_1, torch.zeros(1).to(adv_img_1.device), 
                                    args.epsilon, 0, args.constraint)
            with torch.no_grad():
                adv_img_2 = self.model[1](ori_img_2)
            adv_img_2 = update_and_clip(ori_img_2, adv_img_2, torch.zeros(1).to(adv_img_2.device), 
                                    args.epsilon, 0, args.constraint)
            adv_img[label!=self.ttp_target[0]] += adv_img_1
            adv_img[label==self.ttp_target[0]] += adv_img_2
            return adv_img
        else:
            with torch.no_grad():
                adv_img = self.model[0](ori_img)
            adv_img = update_and_clip(ori_img, adv_img, torch.zeros(1).to(adv_img.device), 
                                    args.epsilon, 0, args.constraint)
            return adv_img