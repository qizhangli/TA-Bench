import os
import random
import torch
import copy
from .helper import update_and_clip
from .newbackend import NewBackend
from .ifgsm import IFGSM

__all__ = ["LGV", "LGVNewBackend"]

class LGV(IFGSM):
    def __init__(self, args, **kwargs):
        self.model = kwargs["source_model"]
        self.model_path = args.model_path
        self._prep()
        self.n_models = args.n_models
        
    def __call__(self, args, ori_img, label, model, verbose=True):
        adv_img = ori_img.clone()
        for i in range(args.steps):
            self.model_list = self._get_model_list_lgv(self.n_models)
            input_grad, loss = self._sample_attack_get_grad(args, adv_img, ori_img, label, i)
            adv_img = update_and_clip(ori_img, adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            if verbose:
                print('Iter {}, Loss {:.4f}'.format(i, loss))
        return adv_img
    
    def _prep(self,):
        self.lgv_model_list = []
        for model_ind in range(40):
            model_cur = copy.deepcopy(self.model)
            state_dict = torch.load(os.path.join(self.model_path, "collect_{}.pt".format(model_ind)))["state_dict"]
            for key in list(state_dict.keys()):
                if key[:6] == "module":
                    state_dict[key[7:]] = state_dict.pop(key)
            model_dict = model_cur.module[1].load_state_dict(state_dict)
            model_cur.eval()
            self.lgv_model_list.append(model_cur)
    
    def _sample_attack_get_grad(self, args, adv_img, ori_img, label, i):
        final_grad, loss_avg = 0, 0
        for model in self.model_list:
            input_grad_cur, loss = self.one_step(args, model, adv_img, ori_img, label, i)
            final_grad += input_grad_cur.data
            loss_avg += loss
        final_grad /= len(self.model_list)
        loss_avg /= len(self.model_list)
        return final_grad, loss_avg
    
    def _get_model_list_lgv(self, n_models):
        self.model_list = []
        for model_ind in range(n_models):
            self.model_list.append(random.choice(self.lgv_model_list))
        return self.model_list


class LGVNewBackend(LGV, NewBackend):
    def __init__(self, args, **kwargs):
        NewBackend.__init__(self, args, **kwargs)
        LGV.__init__(self, args, **kwargs)

    def one_step(self, args, model, adv_img, ori_img, label, i):
        return NewBackend.one_step(self, args, model, adv_img, ori_img, label, i)