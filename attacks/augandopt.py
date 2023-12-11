import numpy as np
import torch
import torch.nn.functional as F
import random
import torchvision.transforms as T
from .helper import update_and_clip
import scipy.stats as st

__all__ = ["AugandOpt"]

class AugandOpt(object):
    def __init__(self, args, **kwargs):
        
        self.combine = args.combine
        self.di_resize_rate = args.di_resize_rate
        self.di_diversity_prob = args.di_diversity_prob
        self.si_m = args.si_m
        self.admix_m = args.admix_m
        self.admix_eta = args.admix_eta
        if args.ti_kernel:
            self.ti_kernel = self.gkern(int(args.ti_kernel.split("_")[1]), int(args.ti_kernel.split("_")[2])).cuda()
        else:
            self.ti_kernel = args.ti_kernel
        self.ti_len = args.ti_len
        self.ps = args.ps
        self.ps_npatch = args.ps_npatch
        self.pre_grad = args.pre_grad
        self.mi_mu, self.ni_mu = args.mi_mu, args.ni_mu
        self.UN, self.pgd = args.UN, args.pgd

    def addnoise(self, x, eps, constraint):
        if constraint == "linf":
            return x + x.new(x.size()).uniform_(-eps, eps)
        elif constraint == "l2":
            return x + eps * F.normalize(x.new(x.size()).uniform_(-1, 1), dim=(1,2,3))
        else:
            raise RuntimeError("unsupport constraint")
    
    def gkern(self, kernlen=7, nsig=3):
        """Returns a 2D Gaussian kernel array."""

        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        
        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = torch.from_numpy(stack_kernel).float()[:, None, :, :]
        return stack_kernel
    
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
    
    def mix(self, x, times):
        if self.admix_eta == 0:
            assert self.admix_m == 1
        return torch.stack([x + self.admix_eta * x[torch.randperm(len(x))] for _ in range(times)])
    
    def translate_c(self, x):
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
    
    def __call__(self, args, ori_img, label, model, verbose=True):
        momentum = 0
        pre_grad = 0
        adv_img = self.addnoise(ori_img, args.epsilon, args.constraint) if self.pgd else ori_img.clone()
        for i in range(args.steps):
            adv_mix_m = self.mix(adv_img, self.admix_m) if self.admix_eta != 0 else [adv_img.clone()]
            input_grad, loss_avg = 0, 0
            for admix_j in torch.arange(self.admix_m):
                adv_img_j = adv_mix_m[admix_j]
                si_j = random.randint(0, self.si_m-1) if self.si_m-1 != 0 else 0
                adv_img_j.requires_grad_(True)
                if self.pre_grad:
                    adv_img_j = adv_img_j + self.ni_mu*args.step_size*pre_grad
                else:
                    adv_img_j = adv_img_j + self.ni_mu*args.step_size*momentum
                if self.ps:
                    adv_img_j = ori_img.data+adv_img_j-adv_img.data + self.sample_for_interaction(adv_img.data - ori_img.data, self.ps_npatch,
                                                                                                        16, ori_img.size(-1),
                                                                                                        1)[0]
                adv_img_j = self.addnoise(adv_img_j, args.epsilon, args.constraint) if self.UN else adv_img_j
                adv_img_j_ = self.input_diversity(self.translate_c(adv_img_j))
                loss = F.cross_entropy(model(adv_img_j_), label, reduction="mean")
                input_grad += torch.autograd.grad(loss, adv_img_j)[0].data
                loss_avg += loss
            input_grad /= (self.si_m*self.admix_m)
            loss_avg /= (self.si_m*self.admix_m)
            if self.ti_kernel is not None:
                input_grad = F.conv2d(input_grad, weight=self.ti_kernel, stride=1, padding=(self.ti_kernel.shape[-1] - 1) // 2, groups=3)
            pre_grad = F.normalize(input_grad, p=1, dim=(1,2,3))
            if self.mi_mu:
                input_grad = self.mi_mu * momentum + pre_grad
                momentum = input_grad
            adv_img = update_and_clip(ori_img, adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            if verbose:
                print("Iter {}, Loss {:.4f}".format(i, loss_avg.item()))
        return adv_img
