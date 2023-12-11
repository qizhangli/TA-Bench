import torch
import torch.nn.functional as F

from .helper import update_and_clip
from .intermediate import ILOutHook
from .newbackend import NewBackend

__all__ = ["FDA", "FDANewBackend"]


def fda_loss(ori_ilout_ls, adv_ilout_ls):
    loss = 0
    for ori_ilout, adv_ilout in zip(ori_ilout_ls, adv_ilout_ls):
        N, C = ori_ilout.size(0), ori_ilout.size(1)
        N_ele = ori_ilout.numel() / N
        ori_ilout_ = ori_ilout.reshape(N, C, -1)
        adv_ilout_ = adv_ilout.reshape(N, C, -1)
        mean_map = ori_ilout_.data.mean(1, keepdim=True)
        good = (ori_ilout_  < mean_map).float()
        bad  = (ori_ilout_ >= mean_map).float()
        loss_  = torch.log((good * adv_ilout_ / N_ele ).norm(p=2, dim=(1,2)))
        loss_ -= torch.log((bad  * adv_ilout_ / N_ele ).norm(p=2, dim=(1,2))) 
        loss += loss_
    return loss.mean()
    
class FDA(ILOutHook):
    def __init__(self, args, **kwargs):
        ILOutHook.__init__(self, args.model_name, args.il_pos)
        self.model = kwargs["source_model"]
        self._prep_hook()

    def __call__(self, args, ori_img, label, model, verbose=True):
        with torch.no_grad():
            self.model(ori_img)
            ori_ilout_ls = self._get_ilout()
        adv_img = ori_img.clone()
        for i in range(args.steps):
            adv_img.requires_grad_(True)
            logits_adv = self.model(adv_img)
            adv_ilout_ls = self._get_ilout()
            loss_ce = F.cross_entropy(logits_adv.data, label)
            loss = fda_loss(ori_ilout_ls, adv_ilout_ls)
            input_grad = torch.autograd.grad(loss, adv_img)[0].data
            adv_img = update_and_clip(ori_img, adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            if verbose:
                print("Iter {}, Loss CE {:.4f}, Loss {:.4f}".format(i, loss_ce.item(), loss.item()))
        return adv_img
            
class FDANewBackend(NewBackend, FDA):
    def __init__(self, args, **kwargs):
        NewBackend.__init__(self, args, **kwargs)
        FDA.__init__(self, args, **kwargs)
    
    def __call__(self, args, ori_img, label, model, verbose=True):
        with torch.no_grad():
            self.model(ori_img)
            ori_ilout_ls = self._get_ilout()
        adv_img = ori_img.clone()
        for i in range(args.steps):
            input_grad, loss_avg = 0, 0
            for j in range(args.aug_times):
                adv_img.requires_grad_(True)
                adv_img_aug = self.get_aug_input(args, adv_img, ori_img, i)
                logits_adv = self.model(adv_img_aug)
                adv_ilout_ls = self._get_ilout()
                loss_ce = F.cross_entropy(logits_adv.data, label)
                loss = fda_loss(ori_ilout_ls, adv_ilout_ls)
                input_grad += torch.autograd.grad(loss, adv_img)[0].data
            input_grad /= args.aug_times
            input_grad = self.get_input_grad(input_grad)
            adv_img = update_and_clip(ori_img, adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            if verbose:
                print("Iter {}, Loss CE {:.4f}, Loss {:.4f}".format(i, loss_ce.item(), loss.item()))
        return adv_img