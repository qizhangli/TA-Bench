import torch
import torch.nn.functional as F

from .helper import update_and_clip

from .intermediate import ILOutHook
from .newbackend import NewBackend


__all__ = ["ILA", "ILANewBackend"]


def ila_loss(guide, ildiff):
    return (ildiff * guide).sum()
    
class ILA(ILOutHook):
    def __init__(self, args, **kwargs):
        ILOutHook.__init__(self, args.model_name, args.il_pos)
        self.model = kwargs["source_model"]
        self.base_steps = args.base_steps

    def __call__(self, args, ori_img, label, model, verbose=True):
        guide, ori_ilout = self._first_stage(args, ori_img, label, model, verbose)
        adv_img = ori_img.clone()
        adv_img = adv_img + 0.001 * torch.randn(adv_img.size()).to(adv_img.device)
        for i in range(args.steps):
            adv_img.requires_grad_(True)
            logits = self.model(adv_img)
            ildiff = self._get_ilout() - ori_ilout
            loss = ila_loss(guide, ildiff) / len(ori_img)
            input_grad = torch.autograd.grad(loss, adv_img)[0].data
            adv_img = update_and_clip(ori_img, adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            if verbose:
                print("Iter {}, Loss CE {:.4f}, Loss {:.4f}".format(i, F.cross_entropy(logits.data, label), loss.item()))
        return adv_img
    
    def _first_stage(self, args, ori_img, label, model, verbose=True):
        base_adv_img = ori_img.clone()
        for i in range(self.base_steps):
            base_adv_img.requires_grad_(True)
            loss = F.cross_entropy(self.model(base_adv_img), label, reduction="mean")
            input_grad = torch.autograd.grad(loss, base_adv_img)[0].data
            base_adv_img = update_and_clip(ori_img, base_adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            if verbose:
                print("First Stage, Iter {}, Loss {:.4f}".format(i, loss.item()))
        self._prep_hook()
        with torch.no_grad():
            self.model(ori_img)
            ori_ilout = self._get_ilout().data
            self.model(base_adv_img)
            baseadv_ilout = self._get_ilout().data
            guide = baseadv_ilout - ori_ilout
            guide = F.normalize(guide, p=2, dim=list(range(len(guide.shape)))[1:]) # optional
        return guide, ori_ilout
            
class ILANewBackend(ILA, NewBackend):
    def __init__(self, args, **kwargs):
        NewBackend.__init__(self, args, **kwargs)
        ILA.__init__(self, args, **kwargs)
    
    def _first_stage(self, args, ori_img, label, model, verbose=True):
        base_adv_img = ori_img.clone()
        for i in range(self.base_steps):
            input_grad, loss_avg = 0, 0
            for j in range(args.aug_times):
                base_adv_img.requires_grad_(True)
                base_adv_img_aug = self.get_aug_input(args, base_adv_img, ori_img, i)
                loss = F.cross_entropy(self.model(base_adv_img_aug), label, reduction="mean")
                input_grad += torch.autograd.grad(loss, base_adv_img)[0].data
            input_grad /= args.aug_times
            input_grad = self.get_input_grad(input_grad)
            base_adv_img = update_and_clip(ori_img, base_adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            if verbose:
                print("First Stage, Iter {}, Loss {:.4f}".format(i, loss.item()))
        self._prep_hook()
        with torch.no_grad():
            self.model(ori_img)
            ori_ilout = self._get_ilout()
            loss = F.cross_entropy(self.model(base_adv_img), label)
            baseadv_ilout = self._get_ilout()
            guide = baseadv_ilout - ori_ilout
            guide = F.normalize(guide, p=2, dim=list(range(len(guide.shape)))[1:]) # optional
        return guide, ori_ilout