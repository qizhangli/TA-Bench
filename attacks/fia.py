
import numpy as np
import torch
import torch.nn.functional as F

from .helper import update_and_clip
from .intermediate import ILOutHook
from .newbackend import NewBackend


__all__ = ["FIA", "FIANewBackend"]

def fia_loss(adv_ilout, delta):
    return (adv_ilout * delta).sum() / len(adv_ilout)

class FIA(ILOutHook):
    def __init__(self, args, **kwargs):
        ILOutHook.__init__(self, args.model_name, args.il_pos)
        self.model = kwargs["source_model"]
        self.n_ens = args.n_ens
        self.p_drop = args.p_drop
        self._prep_hook()
        
    def __call__(self, args, ori_img, label, model, verbose=True):
        delta = self._get_delta(ori_img, label)
        adv_img = ori_img.clone()
        for i in range(args.steps):
            adv_img.requires_grad_(True)
            logits_adv = self.model(adv_img)
            adv_ilout = self._get_ilout()
            loss_ce = F.cross_entropy(logits_adv.data, label)
            loss = fia_loss(adv_ilout, delta)
            input_grad = torch.autograd.grad(loss, adv_img)[0].data
            adv_img = update_and_clip(ori_img, adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            if verbose:
                print("Iter {}, Loss CE {:.4f}, Loss {:.4f}".format(i, loss_ce.item(), loss.item()))
        return adv_img
    
    def _get_delta(self, ori_img, label):
        delta = 0
        for i in range(self.n_ens):
            mask = np.random.binomial(1, self.p_drop, ori_img.size())
            logits = self.model(torch.from_numpy(mask).to(ori_img.device) * ori_img)
            loss = F.cross_entropy(logits, label)
            delta += self._get_ilout_grad(loss)
        delta = F.normalize(delta, p=2, dim=list(range(len(delta.shape)))[1:])
        return delta
    
class FIANewBackend(FIA, NewBackend):
    def __init__(self, args, **kwargs):
        NewBackend.__init__(self, args, **kwargs)
        FIA.__init__(self, args, **kwargs)
        
    def __call__(self, args, ori_img, label, model, verbose=True):
        delta = self._get_delta(ori_img, label)
        adv_img = ori_img.clone()
        for i in range(args.steps):
            input_grad = 0
            for j in range(args.aug_times):
                adv_img.requires_grad_(True)
                adv_img_aug = self.get_aug_input(args, adv_img, ori_img, i)
                logits_adv = self.model(adv_img_aug)
                adv_ilout = self._get_ilout()
                loss_ce = F.cross_entropy(logits_adv.data, label, reduction="mean")
                loss = fia_loss(adv_ilout, delta)
                input_grad += torch.autograd.grad(loss, adv_img)[0].data
            input_grad /= args.aug_times
            input_grad = self.get_input_grad(input_grad)
            adv_img = update_and_clip(ori_img, adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            if verbose:
                print("Iter {}, Loss CE {:.4f}, Loss {:.4f}".format(i, loss_ce.item(), loss.item()))
        return adv_img