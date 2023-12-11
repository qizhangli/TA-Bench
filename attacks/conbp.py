import torch.nn.functional as F

from .linbp import LinBPAct, LinBP
from .newbackend import NewBackend

__all__ = ["ConBP", "ConBPNewBackend"]

class ConBPAct(LinBPAct):
    def forward(self, x):
        x_dist = self.distribute(x)
        x_normal = self.act(x_dist[0])
        x_1 = x_dist[1]
        x_1_ = x_1 + 0
        x_conbp = self.act(x).data - F.softplus(x_1_, 0.5).data + F.softplus(x_1_, 0.5)
        x_out = x_normal + x_conbp
        x_out = x_out - 0.5*x_out.data
        return x_out

class ConBP(LinBP):
    def __init__(self, args, **kwargs):
        self.model = kwargs["source_model"]
        self.pos = args.pos
        self.new_act = ConBPAct
        self._prep(args.model_name)

class ConBPNewBackend(NewBackend, ConBP):
    def __init__(self, args, **kwargs):
        NewBackend.__init__(self, args, **kwargs)
        ConBP.__init__(self, args, **kwargs)
        