import torch
import torch.nn.functional as F

from .helper import update_and_clip

__all__ = ["IFGSM"]

class IFGSM(object):
    def __init__(self, args, **kwargs):
        pass
    
    def one_step(self, args, model, adv_img, ori_img, label, step_ind):
        adv_img.requires_grad_(True)
        loss = F.cross_entropy(model(adv_img), label, reduction="mean")
        input_grad = torch.autograd.grad(loss, adv_img)[0].data
        return input_grad, loss
    
    def __call__(self, args, ori_img, label, model, verbose=True):
        adv_img = ori_img.clone()
        for i in range(args.steps):
            adv_img.requires_grad_(True)
            input_grad, loss = self.one_step(args, model, adv_img, ori_img, label, i)
            adv_img = update_and_clip(ori_img, adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            if verbose:
                print("Iter {}, Loss {:.4f}".format(i, loss.item()))
        return adv_img
