import torch
from .newbackend import NewBackend
import dill
from .ifgsm import IFGSM

__all__ = ["RFA", "RFANewBackend"]

class RFA(IFGSM):
    def __init__(self, args, **kwargs):
        self.model = kwargs["source_model"]
        self._prep_model(args.model_path)
        
    def _prep_model(self, model_path):
        state_dict = torch.load(model_path, pickle_module=dill)
        sd = state_dict['model']
        for key in list(sd.keys()):
            if 'attacker.' in key:
                del sd[key]
            elif 'module.model.' in key:
                sd[key.replace('module.model.','')] = sd[key]
                del sd[key]
            elif 'module.normalizer.' in key:
                del sd[key]
        model_dict = self.model.module[1].load_state_dict(sd)
        self.model.eval()
        
class RFANewBackend(NewBackend, RFA):
    def __init__(self, args, **kwargs):
        NewBackend.__init__(self, args, **kwargs)
        RFA.__init__(self, args, **kwargs)
        