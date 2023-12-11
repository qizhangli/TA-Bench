import torch
import torch.nn.functional as F

from .helper import update_and_clip
from .intermediate import ILOutHook
from .newbackend import NewBackend

__all__ = ["TAP", "TAPNewBackend"]


class TAP(ILOutHook):
    def __init__(self, args, **kwargs):
        ILOutHook.__init__(self, args.model_name, args.il_pos)
        self.model = kwargs["source_model"]
        self.lam = args.lam
        self.eta = args.eta
        self.alpha = args.alpha
        self.ks = args.ks
        self._prep_hook()
            
    def __call__(self, args, ori_img, label, model, verbose=True):
        adv_img = ori_img.clone()
        with torch.no_grad():
            self.model(ori_img)
            ori_ilout_ls = self._get_ilout()
        for i in range(args.steps):
            adv_img.requires_grad_(True)
            logit = self.model(adv_img)
            ilout_ls = self._get_ilout()
            loss, ce_loss = self._tap_loss(adv_img, ori_img, label, ilout_ls, logit, ori_ilout_ls)
            input_grad = torch.autograd.grad(loss, adv_img)[0].data
            adv_img = update_and_clip(ori_img, adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            if verbose:
                print("Iter {}, Loss CE {:.4f}".format(i, ce_loss))
        return adv_img
    
    def _tap_loss(self, adv_img, ori_img, label, ilout_ls, logit, ori_ilout_ls):
        l1 = F.cross_entropy(logit, label)
        l2 = 0
        for i, ilout in enumerate(ilout_ls):
            a = torch.sign(ori_ilout_ls[i]) * torch.pow(torch.abs(ori_ilout_ls[i])+1e-12, self.alpha)
            b = torch.sign(ilout) * torch.pow(torch.abs(ilout)+1e-12, self.alpha)
            l2 += self.lam * ((a - b).norm(p=2, dim=list(range(len(a.shape)))[1:]) ** 2).sum() / len(a)
        l3 = self.eta * torch.abs(F.avg_pool2d(ori_img - adv_img, self.ks)).sum() / len(a)
        return l1 + l2 + l3, l1.item()


class TAPNewBackend(TAP, NewBackend):
    def __init__(self, args, **kwargs):
        NewBackend.__init__(self, args, **kwargs)
        TAP.__init__(self, args, **kwargs)
        
    def __call__(self, args, ori_img, label, model, verbose=True):
        adv_img = ori_img.clone()
        with torch.no_grad():
            self.model(ori_img)
            ori_ilout_ls = self._get_ilout()
        for i in range(args.steps):
            input_grad = 0
            for j in range(args.aug_times):
                adv_img.requires_grad_(True)
                adv_img_aug = self.get_aug_input(args, adv_img, ori_img, i)
                logit = self.model(adv_img_aug)
                ilout_ls = self._get_ilout()
                loss, ce_loss = self._tap_loss(adv_img, ori_img, label, ilout_ls, logit, ori_ilout_ls)
                input_grad += torch.autograd.grad(loss, adv_img)[0].data
            input_grad /= args.aug_times
            input_grad = self.get_input_grad(input_grad)
            adv_img = update_and_clip(ori_img, adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            if verbose:
                print("Iter {}, Loss CE {:.4f}".format(i, ce_loss))
        return adv_img
    