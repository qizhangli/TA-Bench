import torch
from .newbackend import NewBackend
from .ifgsm import IFGSM

__all__ = ["DRA", "DRANewBackend"]

class DRA(IFGSM):
    def __init__(self, args, **kwargs):
        self.model = kwargs["source_model"]
        self._prep_model(args.model_path)
        
    def _prep_model(self, model_path):
        state_dict = torch.load(model_path)["model_state_dict"]
        state_dict["module.fc.weight"] = state_dict.pop("module.last_linear.weight")
        state_dict["module.fc.bias"] = state_dict.pop("module.last_linear.bias")
        for key in list(state_dict.keys()):
            if key[:6] == "module":
                state_dict[key[7:]] = state_dict.pop(key)
        self.model.module[1].load_state_dict(state_dict)
        self.model.eval()
        
class DRANewBackend(NewBackend, DRA):
    def __init__(self, args, **kwargs):
        NewBackend.__init__(self, args, **kwargs)
        DRA.__init__(self, args, **kwargs)
        