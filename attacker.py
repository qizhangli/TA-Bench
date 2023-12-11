import os

import numpy as np
import torch

from attacks.helper import to_np_uint8, update_and_clip
from attacks import *

class Attacker(object):
    def __init__(self, args, source_model, dataloader):
        super(Attacker, self).__init__()
        self.source_model = source_model
        self.dataloader = dataloader
        self.params = args
        self.attack_func = eval(args.method)(args, source_model=source_model)
        
    def attack(self, verbose=True):
        label_ls = []
        for ind, (ori_img, label) in enumerate(self.dataloader):
            label_ls.append(label)
            ori_img, label = ori_img.cuda(), label.cuda()
            adv_img = self.attack_func(args=self.params, ori_img=ori_img, label=label, model=self.source_model, verbose=verbose)
            np.save(os.path.join(self.params.save_dir, 'batch_{}.npy'.format(ind)), to_np_uint8(adv_img))
            print(' batch_{}.npy saved'.format(ind))
        label_ls = torch.cat(label_ls)
        np.save(os.path.join(self.params.save_dir, 'labels.npy'), label_ls.numpy())
        print('DONE.')
        