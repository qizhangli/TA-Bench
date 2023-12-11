import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from .helper import update_and_clip

__all__ = ["NewBackend"]

class NewBackend(object):
    def __init__(self, args, **kwargs):
        self.di_resize_rate = args.di_resize_rate
        self.di_diversity_prob = args.di_diversity_prob
        self.ti_len = args.ti_len
        self.ps_npatch = args.ps_npatch
        self.mi_mu, self.ni_mu = args.mi_mu, args.ni_mu
        self.constraint = args.constraint
        
    def addnoise(self, x, eps):
        if self.constraint == "linf":
            return x + x.new(x.size()).uniform_(-eps, eps)
        elif self.constraint == "l2":
                return x + eps * F.normalize(x.new(x.size()).uniform_(-1, 1), dim=(1,2,3))
        else:
            raise RuntimeError("unsupport constraint")
    
    def input_diversity(self, x):
        if torch.rand(1) >= self.di_diversity_prob:
            return x
        img_size = x.shape[-1]
        img_resize = int(img_size * self.di_resize_rate)
        if self.di_resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]
        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded
    
    def translate(self, x):
        if self.ti_len == 0:
            return x
        return T.RandomAffine(0, translate=((self.ti_len+1) / x.shape[-1], (self.ti_len+1) / x.shape[-2]))(x)
    
    def sample_grids(self, sample_grid_num=16,
                 grid_scale=16,
                 img_size=224,
                 sample_times=8):
        grid_size = img_size // grid_scale
        sample = []
        for _ in range(sample_times):
            grids = []
            ids = np.random.randint(0, grid_scale**2, size=sample_grid_num)
            rows, cols = ids // grid_scale, ids % grid_scale
            for r, c in zip(rows, cols):
                grid_range = (slice(r * grid_size, (r + 1) * grid_size),
                            slice(c * grid_size, (c + 1) * grid_size))
                grids.append(grid_range)
            sample.append(grids)
        return sample

    def sample_for_interaction(self, delta,
                            sample_grid_num,
                            grid_scale,
                            img_size,
                            times=16):
        samples = self.sample_grids(
            sample_grid_num=sample_grid_num,
            grid_scale=grid_scale,
            img_size=img_size,
            sample_times=times)
        only_add_one_mask = torch.zeros_like(delta)[None, :, :, :, :].repeat(times, 1, 1, 1, 1)
        for i in range(times):
            grids = samples[i]
            for grid in grids:
                only_add_one_mask[i:i + 1, :, :, grid[0], grid[1]] = 1
        only_add_one_perturbation = delta * only_add_one_mask
        return only_add_one_perturbation
    
    def get_aug_input(self, args, adv_img, ori_img, step_ind):
        if step_ind == 0:
            self.momentum = 0
        adv_img_aug = adv_img + self.ni_mu*args.step_size*self.momentum
        adv_img_aug = adv_img_aug - (adv_img.data - ori_img.data) + self.sample_for_interaction(adv_img.data - ori_img.data, 
                                                                                            self.ps_npatch,
                                                                                            16, 
                                                                                            ori_img.size(-1),
                                                                                            1)[0]
        adv_img_aug = self.addnoise(adv_img_aug, args.epsilon)
        adv_img_aug = self.input_diversity(self.translate(adv_img_aug))
        return adv_img_aug
    
    def get_input_grad(self, input_grad):
        self.pre_grad = F.normalize(input_grad, p=1, dim=(1,2,3))
        input_grad = self.mi_mu * self.momentum + self.pre_grad
        self.momentum = input_grad
        return input_grad
    
    def one_step(self, args, model, adv_img, ori_img, label, step_ind):
        input_grad, loss_avg = 0, 0
        for j in range(args.aug_times):
            adv_img.requires_grad_(True)
            adv_img_aug = self.get_aug_input(args, adv_img, ori_img, step_ind)
            loss = F.cross_entropy(model(adv_img_aug), label, reduction="mean")
            input_grad += torch.autograd.grad(loss, adv_img)[0].data
            loss_avg += loss.item()
        loss_avg /= args.aug_times
        input_grad /= args.aug_times
        input_grad = self.get_input_grad(input_grad)
        return input_grad, loss_avg
    
    def __call__(self, args, ori_img, label, model, verbose=True):
        adv_img = ori_img.clone()
        for i in range(args.steps):
            input_grad, loss = self.one_step(args, model, adv_img, ori_img, label, i)
            adv_img = update_and_clip(ori_img, adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            if verbose:
                print("Iter {}, Loss {:.4f}".format(i, loss))
        return adv_img
