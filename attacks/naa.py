import torch
import torch.nn.functional as F

from .helper import update_and_clip
from .intermediate import ILOutHook
from .newbackend import NewBackend

__all__ = ["NAA", "NAANewBackend"]


def naa_loss(ori_ilout, adv_ilout, weights, gamma):
    attribution = (adv_ilout-ori_ilout)*weights
    pos = (attribution >= 0).float()
    balance_attribution = pos * attribution + gamma * (1-pos) * attribution 
    return balance_attribution.sum() / len(ori_ilout)
    
class NAA(ILOutHook):
    def __init__(self, args, **kwargs):
        ILOutHook.__init__(self, args.model_name, args.il_pos)
        self.model = kwargs["source_model"]
        self.n_ens = args.n_ens
        self.gamma = args.gamma
        self._prep_hook()
        
    def __call__(self, args, ori_img, label, model, verbose=True):
        IA, ori_ilout = self._get_IA(ori_img, label)
        adv_img = ori_img.clone()
        for i in range(args.steps):
            adv_img.requires_grad_(True)
            logits_adv = self.model(adv_img)
            adv_ilout = self._get_ilout()
            loss_ce = F.cross_entropy(logits_adv.data, label)
            loss = naa_loss(ori_ilout, adv_ilout, IA, self.gamma)
            input_grad = torch.autograd.grad(loss, adv_img)[0].data
            adv_img = update_and_clip(ori_img, adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            if verbose:
                print("Iter {}, Loss CE {:.4f}, Loss {:.4f}".format(i, loss_ce.item(), loss.item()))
        return adv_img
    
    def _get_IA(self, ori_img, label):
        IA = 0
        for i in range(self.n_ens):
            x_base = 0
            images_tmp2 = ori_img + 0.2 * torch.randn(ori_img.size()).to(ori_img.device)
            images_tmp2 = images_tmp2*(1-i/self.n_ens)+(i/self.n_ens)*x_base
            logits = self.model(images_tmp2)
            loss = F.cross_entropy(logits, label)
            IA += self._get_ilout_grad(loss)
        logits = self.model(ori_img)
        loss = F.cross_entropy(logits, label)
        ori_ilout = self._get_ilout(clear=False).data
        IA += self._get_ilout_grad(loss)
        IA = F.normalize(IA, p=2, dim=list(range(len(IA.shape)))[1:])
        return IA, ori_ilout
    
class NAANewBackend(NAA, NewBackend):
    def __init__(self, args, **kwargs):
        NewBackend.__init__(self, args, **kwargs)
        NAA.__init__(self, args, **kwargs)
        
    def __call__(self, args, ori_img, label, model, verbose=True):
        IA, ori_ilout = self._get_IA(ori_img, label)
        adv_img = ori_img.clone()
        for i in range(args.steps):
            input_grad, loss_avg = 0, 0
            for j in range(args.aug_times):
                adv_img.requires_grad_(True)
                adv_img_aug = self.get_aug_input(args, adv_img, ori_img, i)
                logits_adv = self.model(adv_img_aug)
                adv_ilout = self._get_ilout()
                loss_ce = F.cross_entropy(logits_adv.data, label)
                loss = naa_loss(ori_ilout, adv_ilout, IA, self.gamma)
                input_grad += torch.autograd.grad(loss, adv_img)[0].data
            input_grad /= args.aug_times
            input_grad = self.get_input_grad(input_grad)
            adv_img = update_and_clip(ori_img, adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            if verbose:
                print("Iter {}, ce loss {:.4f}, loss {:.4f}".format(i, loss_ce.item(), loss.item()))
        return adv_img