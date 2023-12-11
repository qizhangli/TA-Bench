import copy
import numpy as np
import torch
import torch.nn.functional as F

from .helper import update_and_clip

__all__ = ["IR", "IRBaseline"]


def interaction_loss(outputs, leave_one_outputs, 
                     only_add_one_outputs, zero_outputs,
                     target, label):
    arange_n = torch.arange(len(outputs))
    arange_n_r = torch.arange(len(leave_one_outputs))
    label_r = label.repeat(len(leave_one_outputs) // len(outputs))
    target_r = target.repeat(len(leave_one_outputs) // len(outputs))
    complete_score = (outputs[arange_n, target] - outputs[arange_n, label]).sum() / len(outputs)
    leave_one_out_score = (
        leave_one_outputs[arange_n_r, target_r] -
        leave_one_outputs[arange_n_r, label_r]).sum() / len(leave_one_outputs)
    only_add_one_score = (
        only_add_one_outputs[arange_n_r, target_r] -
        only_add_one_outputs[arange_n_r, label_r]).sum() / len(leave_one_outputs)
    average_pairwise_interaction = (complete_score - leave_one_out_score -
                                    only_add_one_score + 0 )
    return average_pairwise_interaction

def sample_grids(sample_grid_num=16,
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


def sample_for_interaction(delta,
                           sample_grid_num,
                           grid_scale,
                           img_size,
                           times=16):
    samples = sample_grids(
        sample_grid_num=sample_grid_num,
        grid_scale=grid_scale,
        img_size=img_size,
        sample_times=times)
    only_add_one_mask = torch.zeros_like(delta)[None, :, :, :, :].repeat(times, 1, 1, 1, 1)
    for i in range(times):
        grids = samples[i]
        for grid in grids:
            only_add_one_mask[i:i + 1, :, :, grid[0], grid[1]] = 1
    leave_one_mask = 1 - only_add_one_mask
    only_add_one_perturbation = delta * only_add_one_mask
    leave_one_out_perturbation = delta * leave_one_mask

    return only_add_one_perturbation, leave_one_out_perturbation

def get_features(
    model,
    x,
    perturbation,
    leave_one_out_perturbation,
    only_add_one_perturbation,
):
    x_r = x[None, :, :, :, :].repeat(len(leave_one_out_perturbation), 1, 1, 1, 1)
    outputs = model(x + perturbation)
    leave_one_outputs = model((x_r + leave_one_out_perturbation).view([-1] + list(x_r.shape[2:])))
    only_add_one_outputs = model((x_r + only_add_one_perturbation).view([-1] + list(x_r.shape[2:])))
    zero_outputs = 0
    return (outputs, leave_one_outputs, only_add_one_outputs, zero_outputs)

class IR(object):
    def __init__(self, args, **kwargs):
        self.sample_grid_num = args.sample_grid_num
        self.grid_scale = args.grid_scale
        self.sample_times = args.sample_times
        self.lam = args.lam
        
    def __call__(self, args, ori_img, label, model, verbose=True):
        batch_size, image_width = ori_img.size(0), ori_img.size(-1)
        adv_img = ori_img.clone()
        for i in range(args.steps):
            adv_img.requires_grad_(True)
            att_output = model(adv_img)
            loss_ce = F.cross_entropy(att_output, label, reduction="mean")
            delta = adv_img - ori_img
            only_add_one_perturbation, leave_one_out_perturbation = \
                sample_for_interaction(delta, self.sample_grid_num,
                                        self.grid_scale, image_width,
                                        self.sample_times)
            (outputs, leave_one_outputs, only_add_one_outputs,
                zero_outputs) = get_features(model, ori_img, delta,
                                            leave_one_out_perturbation,
                                            only_add_one_perturbation)
            outputs_c = copy.deepcopy(outputs.detach())
            outputs_c[torch.arange(batch_size), label] = -np.inf
            other_max = outputs_c.argmax(1)
            average_pairwise_interaction = interaction_loss(
                outputs, leave_one_outputs, only_add_one_outputs,
                zero_outputs, other_max, label)
            loss_interaction = -self.lam * average_pairwise_interaction
            loss = loss_ce + loss_interaction
            input_grad = torch.autograd.grad(loss, adv_img)[0].data
            adv_img = update_and_clip(ori_img, adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            if verbose:
                print("Iter {}, Loss CE {:.4f}, Loss {:.4f}".format(i, loss_ce.item(), loss.item()))
        return adv_img


def sample_aug_perturbation(delta,
                           sample_grid_num,
                           grid_scale,
                           img_size,
                           times=16):
    samples = sample_grids(
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


class IRBaseline(object):
    def __init__(self, args, **kwargs):
        self.sample_grid_num = args.sample_grid_num
        self.grid_scale = args.grid_scale
        self.sample_times = args.sample_times
        
    def __call__(self, args, ori_img, label, model, verbose=True):
        adv_img = ori_img.clone()
        for i in range(args.steps):
            input_grad = 0
            for j in range(self.sample_times):
                adv_img.requires_grad_(True)
                ttt = sample_aug_perturbation(adv_img.data-ori_img, self.sample_grid_num,
                                            self.grid_scale, ori_img.size(-1),
                                            1)[0]
                att_output = model(adv_img - adv_img.data + ori_img + ttt)
                loss_ce = F.cross_entropy(att_output, label, reduction="mean")
                loss = loss_ce
                input_grad += torch.autograd.grad(loss, adv_img)[0].data
            input_grad /= self.sample_times
            adv_img = update_and_clip(ori_img, adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            if verbose:
                print("Iter {}, Loss {:.4f}".format(i, loss_ce.item()))
        return adv_img

