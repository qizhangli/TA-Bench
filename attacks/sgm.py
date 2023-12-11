from .newbackend import NewBackend
from .ifgsm import IFGSM

def _get_sgm_hook(gamma):
    def sgm_hook(module, grad_input, grad_output):
        return (gamma * grad_input[0], )
    return sgm_hook

class SGM(IFGSM):
    def __init__(self, args, **kwargs):
        self.model = kwargs["source_model"]
        self.gamma = args.sgm_gamma
        self.model_name = args.model_name
        self._prep()
    
    def _prep_resnet(self, ):
        for name, module in self.model.named_modules():
            if 'act' in name and not '0.act' in name:
                module.inplace=False
                module.register_full_backward_hook(_get_sgm_hook(self.gamma**0.5))
    
    # for vit, deit, beit, mixer, swin, convnext
    def _prep_deit(self, ):
        for name, module in self.model.named_modules():
            if "drop_path" in name:
                module.register_full_backward_hook(_get_sgm_hook(self.gamma))
        
    # for EffNetV2-M
    def _prep_effnetv2m(self, ):
        print("here")
        for name, module in self.model.named_modules():
            if "_modules" in module.__dict__.keys():
                if "drop_path" in module._modules.keys():
                    if module.__dict__["has_skip"]:
                        module.drop_path.register_full_backward_hook(_get_sgm_hook(self.gamma))
                        
    def _prep(self, ):
        if self.model_name in ["tv_resnet50"]:
            self._prep_resnet()
        elif self.model_name in ["vit_base_patch16_224", "deit3_base_patch16_224", 
                                 "beit_base_patch16_224", "mixer_b16_224", 
                                 "convnext_base", "swin_base_patch4_window7_224"]:
            self._prep_deit()
        elif self.model_name in ["tf_efficientnetv2_m"]:
            self._prep_effnetv2m()
        else:
            raise RuntimeError("not support")


class SGMNewBackend(NewBackend, SGM):
    def __init__(self, args, **kwargs):
        NewBackend.__init__(self, args, **kwargs)
        SGM.__init__(self, args, **kwargs)
        