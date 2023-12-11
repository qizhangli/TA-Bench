import torch
import torch.nn.functional as F

from .helper import update_and_clip
from .intermediate import ILOutHook
from .newbackend import NewBackend

__all__ = ["SE", "SENewBackend"]

class SE(ILOutHook):
    def __init__(self, args, **kwargs):
        ILOutHook.__init__(self, args.model_name, args.il_pos)

        self.model = kwargs["source_model"]
        self._prep_hook()

    def __call__(self, args, ori_img, label, model, verbose=True):
        adv_img = ori_img.clone()
        for i in range(args.steps):
            adv_img.requires_grad_(True)
            logits_adv = self.model(adv_img)
            adv_ilout_ls = self._get_ilout()
            loss, loss_ce = self._get_se_loss(adv_ilout_ls, label)
            input_grad = torch.autograd.grad(loss, adv_img)[0].data
            adv_img = update_and_clip(ori_img, adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            if verbose:
                print("Iter {}, Loss CE {:.4f}, Loss {:.4f}".format(i, loss_ce.item(), loss.item()))
        return adv_img
    
    def _get_se_loss(self, adv_ilout_ls, label):
        loss = 0
        for adv_ilout in adv_ilout_ls:
            if self.model_name == "mixer_b16_224":
                if self.model.module[1].global_pool == 'avg':
                    out = adv_ilout.mean(dim=1)
                out = self.model.module[1].head(out)
            else:
                out = self.model.module[1].forward_head(self.model.module[1].norm(adv_ilout))
            loss_ce = F.cross_entropy(out, label)
            loss += loss_ce
        return loss, loss_ce
    
    
class SENewBackend(SE, NewBackend):
    def __init__(self, args, **kwargs):
        NewBackend.__init__(self, args, **kwargs)
        SE.__init__(self, args, **kwargs)
    
    def __call__(self, args, ori_img, label, model, verbose=True):
        adv_img = ori_img.clone()
        for i in range(args.steps):
            input_grad, loss_avg = 0, 0
            for j in range(args.aug_times):
                adv_img.requires_grad_(True)
                adv_img_aug = self.get_aug_input(args, adv_img, ori_img, i)
                logits_adv = self.model(adv_img_aug)
                adv_ilout_ls = self._get_ilout()
                loss, loss_ce = self._get_se_loss(adv_ilout_ls, label)
                input_grad += torch.autograd.grad(loss, adv_img)[0].data
            input_grad /= args.aug_times
            input_grad = self.get_input_grad(input_grad)
            adv_img = update_and_clip(ori_img, adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            if verbose:
                print("Iter {}, ce loss {:.4f}, loss {:.4f}".format(i, loss_ce.item(), loss.item()))
        return adv_img