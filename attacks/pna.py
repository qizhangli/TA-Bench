from .newbackend import NewBackend

__all__ = ["PNA", "PNANewBackend"]

def pna_hook(module, grad_input, grad_output):
    return (0*grad_input[0], )

class PNA(object):
    def __init__(self, args, **kwargs):
        self.model = kwargs["source_model"]
        self.model_name = args.model_name
        self._prep()
    
    def _prep(self, ):
        if self.model_name in ["vit_base_patch16_224", "deit3_base_patch16_224", 
                                 "beit_base_patch16_224", "swin_base_patch4_window7_224"]:
            for name, module in self.model.named_modules():
                if name[-4:] == "attn":
                    module.attn_drop.register_full_backward_hook(pna_hook)
        else:
            raise RuntimeError("not support")


class PNANewBackend(NewBackend, PNA):
    def __init__(self, args, **kwargs):
        NewBackend.__init__(self, args, **kwargs)
        PNA.__init__(self, args, **kwargs)
        