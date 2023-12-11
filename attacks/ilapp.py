import torch
import torch.nn.functional as F

from .helper import update_and_clip
from .ila import ila_loss
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from .intermediate import ILOutHook
from .newbackend import NewBackend

__all__ = ["ILApp", "ILAppNewBackend"]

def parallel_fit(diff, loss, linr = "svr"):
    if linr == "rr":
        clf = Ridge(alpha=1e10, fit_intercept=False, tol=1e-4)
    elif linr == "svr":
        clf = LinearSVR(tol=1e-4, C=1e-10)
    else:
        raise RuntimeError("not support")
    clf.fit(diff, loss)
    return torch.from_numpy(clf.coef_).float()

class ILApp(ILOutHook):
    def __init__(self, args, **kwargs):
        ILOutHook.__init__(self, args.model_name, args.il_pos)
        self.model = kwargs["source_model"]
        self.njobs = args.njobs
        self.rand_restart = args.rand_restart
        self.linr = args.linr
        
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
        ildiff_ls, loss_ls = [], []
        base_adv_img = ori_img.clone()
        self._prep_hook()
        with torch.no_grad():
            self.model(ori_img)
            ori_ilout = self._get_ilout()
            ori_ilout = ori_ilout.data.cpu()
        for i in range(args.base_steps):
            base_adv_img.requires_grad_(True)
            logit = self.model(base_adv_img)
            adv_ilout = self._get_ilout()
            loss_ = F.cross_entropy(logit, label, reduction="none")
            loss = loss_.mean()
            input_grad = torch.autograd.grad(loss, base_adv_img)[0].data
            base_adv_img = update_and_clip(ori_img, base_adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            ildiff_ls.append(adv_ilout.data.cpu() - ori_ilout)
            loss_ls.append(loss_.data.cpu())
            if verbose:
                print("First Stage, Iter {}, Loss {:.4f}".format(i, loss.item()))
        if verbose:
            print("Performing Linear Regression ... ")
        ildiff_ls = torch.stack(ildiff_ls).transpose(0,1).flatten(2)
        loss_ls = torch.stack(loss_ls).transpose(0,1)
        guide = Parallel(n_jobs=self.njobs)(delayed(parallel_fit)(ildiff_ls[f_i].clone(), 
                                                                  loss_ls[f_i].clone(),self.linr) for f_i in range(len(label)))
        del ildiff_ls, loss_ls
        guide = torch.stack(guide).view(ori_ilout.size()).to(ori_img.device)
        guide = F.normalize(guide, p=2, dim=list(range(len(guide.shape)))[1:]) # optional
        ori_ilout = ori_ilout.to(ori_img.device)
        return guide, ori_ilout
    
class ILAppNewBackend(ILApp, NewBackend):
    def __init__(self, args, **kwargs):
        NewBackend.__init__(self, args, **kwargs)
        ILApp.__init__(self, args, **kwargs)
    
    def __call__(self, args, ori_img, label, model, verbose=True):
        guide, ori_ilout = self._first_stage(args, ori_img, label, model, verbose)
        adv_img = ori_img.clone()
        adv_img = adv_img + 0.001 * torch.randn(adv_img.size()).to(adv_img.device)
        for i in range(args.steps):
            input_grad, loss_avg = 0, 0
            for j in range(args.aug_times):
                adv_img.requires_grad_(True)
                adv_img_aug = self.get_aug_input(args, adv_img, ori_img, i)
                logits = self.model(adv_img_aug)
                ildiff = self._get_ilout() - ori_ilout
                loss = ila_loss(guide, ildiff) / len(ori_img)
                input_grad += torch.autograd.grad(loss, adv_img)[0].data
            input_grad /= args.aug_times
            input_grad = self.get_input_grad(input_grad)
            adv_img = update_and_clip(ori_img, adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            if verbose:
                print("Iter {}, Loss CE {:.4f}, Loss {:.4f}".format(i, F.cross_entropy(logits.data, label), loss.item()))
        return adv_img
    
    def _first_stage(self, args, ori_img, label, model, verbose=True):
        ildiff_ls, loss_ls = [], []
        base_adv_img = ori_img.clone()
        self._prep_hook()
        with torch.no_grad():
            self.model(ori_img)
            ori_ilout = self._get_ilout()
            ori_ilout = ori_ilout.data.cpu()
        for i in range(args.base_steps):
            base_adv_img.requires_grad_(True)
            logit = self.model(base_adv_img)
            adv_ilout = self._get_ilout()
            loss_ = F.cross_entropy(logit, label, reduction="none")
            loss = loss_.mean()
            input_grad = torch.autograd.grad(loss, base_adv_img)[0].data
            base_adv_img = update_and_clip(ori_img, base_adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            ildiff_ls.append(adv_ilout.data.cpu() - ori_ilout)
            loss_ls.append(loss_.data.cpu())
            if verbose:
                print("First Stage, Iter {}, Loss {:.4f}".format(i, loss.item()))
        if verbose:
            print("Performing Linear Regression ... ")
        ildiff_ls = torch.stack(ildiff_ls).transpose(0,1).flatten(2)#.numpy()
        loss_ls = torch.stack(loss_ls).transpose(0,1)#.numpy()
        guide = Parallel(n_jobs=self.njobs)(delayed(parallel_fit)(ildiff_ls[f_i].clone(), 
                                                                  loss_ls[f_i].clone(),self.linr) for f_i in range(len(label)))
        del ildiff_ls, loss_ls
        guide = torch.stack(guide).view(ori_ilout.size()).to(ori_img.device)
        guide = F.normalize(guide, p=2, dim=list(range(len(guide.shape)))[1:]) # optional
        ori_ilout = ori_ilout.to(ori_img.device)
        return guide, ori_ilout