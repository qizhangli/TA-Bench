import torch
import copy
from .helper import update_and_clip
from .newbackend import NewBackend
from .ifgsm import IFGSM
from collections import OrderedDict

__all__ = ["MoreBayesian", "MoreBayesianNewBackend"]

def get_grad_cat(grad_dict):
    dls = []
    for name, d in grad_dict.items():
        dls.append(d)
    return torch.cat([x.view(-1) for x in dls])

class MoreBayesian(IFGSM):
    def __init__(self, args, **kwargs):
        self.model = kwargs["source_model"]
        self.model_path = args.model_path
        self.mean_model, self.sqmean_model = self._prep()
        self.n_models = args.n_models
        self.morebayesian_scale=args.morebayesian_scale
        
    def __call__(self, args, ori_img, label, model, verbose=True):
        adv_img = ori_img.clone()
        for i in range(args.steps):
            self.model_list = self._get_model_list_morebayesian(self.n_models)
            input_grad, loss = self._sample_attack_get_grad(args, adv_img, ori_img, label, i)
            adv_img = update_and_clip(ori_img, adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            if verbose:
                print('Iter {}, Loss {:.4f}'.format(i, loss))
        return adv_img
    
    def _prep(self,):
        state_dict = torch.load(self.model_path)
        
        state_dict_mean = state_dict["mean_state_dict"]
        for key in list(state_dict_mean.keys()):
            if key[:6] == "module":
                state_dict_mean[key[7:]] = state_dict_mean.pop(key)
        mean_model = copy.deepcopy(self.model)
        mean_model.module[1].load_state_dict(state_dict_mean)
        
        sqmean_model = copy.deepcopy(self.model)
        state_dict_sqmean = state_dict["sqmean_state_dict"]
        for key in list(state_dict_sqmean.keys()):
            if key[:6] == "module":
                state_dict_sqmean[key[7:]] = state_dict_sqmean.pop(key)
        sqmean_model.module[1].load_state_dict(state_dict_sqmean)
        del state_dict, state_dict_mean, state_dict_sqmean
        return mean_model, sqmean_model

    def _sample_attack_get_grad(self, args, adv_img, ori_img, label, i):
        final_grad, loss_avg = 0, 0
        for model in self.model_list:
            input_grad_cur, loss = self.one_step(args, model, adv_img, ori_img, label, i)
            final_grad += input_grad_cur.data
            loss_avg += loss
        final_grad /= len(self.model_list)
        loss_avg /= len(self.model_list)
        return final_grad, loss_avg
    
    def _get_model_list_morebayesian(self, n_models):
        self.model_list = []
        for model_ind in range(n_models):
            self.model_list.append(copy.deepcopy(self.mean_model))
            var_avg = 0
            c = 0
            noise_dict = OrderedDict()
            for (name, param_mean), param_sqmean, param_cur in zip(self.mean_model.named_parameters(), self.sqmean_model.parameters(), self.model_list[-1].parameters()):
                var = torch.clamp(param_sqmean.data - param_mean.data**2, 1e-30)
                var_avg += var.sum()
                noise_dict[name] = var.sqrt() * torch.randn_like(param_mean, requires_grad=False)
                c += param_mean.numel()
            var_avg = var_avg / c
            noise_norm = get_grad_cat(noise_dict).norm(p=2)
            for (name, param_cur), (_, noise) in zip(self.model_list[-1].named_parameters(), noise_dict.items()):
                param_cur.data.add_(noise, alpha=self.morebayesian_scale)
        return self.model_list


class MoreBayesianNewBackend(MoreBayesian, NewBackend):
    def __init__(self, args, **kwargs):
        NewBackend.__init__(self, args, **kwargs)
        MoreBayesian.__init__(self, args, **kwargs)

    def one_step(self, args, model, adv_img, ori_img, label, i):
        return NewBackend.one_step(self, args, model, adv_img, ori_img, label, i)