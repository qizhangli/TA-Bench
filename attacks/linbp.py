import torch
import torch.nn as nn

from .newbackend import NewBackend
from .ifgsm import IFGSM

__all__ = ["LinBP", "LinBPNewBackend"]

def convert_name(name):
    name_ls = name.split(".")
    new_name = ""
    for ss in name_ls:
        if ss.isnumeric():
            new_name += "[{}]".format(ss)
        else:
            new_name += ".{}".format(ss)
    new_name = new_name[1:]
    new_name.replace("].", "]")
    return new_name


class DistributeMainStream(nn.Module):
    def __init__(self, ):
        super(DistributeMainStream, self).__init__()
        self.register_backward_hook(self._backward_hook())
    
    def forward(self, x):
        return torch.stack([x,x])
    
    def _backward_hook(self, ):
        def _hook(module, grad_in, grad_out):
            return (0*grad_in[0], 
                    (grad_in[0].norm(p=2) / grad_in[1].norm(p=2)) * grad_in[1])
        return _hook


class LinBPAct(nn.Module):
    def __init__(self, act):
        super(LinBPAct, self).__init__()
        self.distribute = DistributeMainStream()
        self.act = act
        if "inplace" in self.act.__dict__.keys():
            self.act.__dict__["inplace"] = False
            
    def forward(self, x):
        x_dist = self.distribute(x)
        x_normal = self.act(x_dist[0])
        x_1 = x_dist[1]
        x_1_ = x_1 + 0
        x_linbp = self.act(x_1).data - x_1_.data + x_1_
        x_out = x_normal + x_linbp
        x_out = x_out - 0.5*x_out.data
        return x_out


# for Inception v3
class BasicConv2dNew(nn.Module):
    def __init__(self, ori_module):
        super(BasicConv2dNew, self).__init__()
        self.ori_module = ori_module
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.ori_module.conv(x)
        x = self.ori_module.bn(x)
        return self.act(x)


class LinBP(IFGSM):
    def __init__(self, args, **kwargs):
        self.model = kwargs["source_model"]
        self.pos = args.pos
        self.new_act = LinBPAct
        self._prep(args.model_name)
        
    def _prep(self, model_name):
        if model_name in ["tf_efficientnetv2_m"]:
            self._prep_effnetv2()
        elif model_name in ["inception_v3"]:
            self._prep_inceptionv3()
        elif model_name in ["tv_resnet50"]:
            self._prep_resnet()
        elif model_name in ["vgg19_bn"]:
            self._prep_vgg()
        elif model_name in ["convnext_base"]:
            self._prep_convnext()
        elif model_name in ["vit_base_patch16_224","deit3_base_patch16_224","beit_base_patch16_224","mixer_b16_224"]:
            self._prep_vit()
        elif model_name in ["swin_base_patch4_window7_224"]:
            self._prep_swin()
        else:
            raise RuntimeError("invalid source model")
    
    def _prep_resnet(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU) and "layer" in name:
                if ("act1" in name or "act2" in name) and int(name.split("layer")[-1].split(".")[0]) >= int(self.pos):
                    new_name = convert_name(name)
                    exec("self.model.{}=self.new_act(module)".format(new_name))
                    
    def _prep_vgg(self):
        for name, module in self.model.named_modules():
            if "features" in name:
                if isinstance(module, nn.ReLU) and int(name.split("features.")[-1]) >= int(self.pos):
                    new_name = convert_name(name)
                    exec("self.model.{}=self.new_act(module)".format(new_name))
            elif "pre_logits" in name:
                if isinstance(module, nn.ReLU):
                    new_name = convert_name(name)
                    exec("self.model.{}=self.new_act(module)".format(new_name))
    
    def _prep_convnext(self):
        for name, module in self.model.named_modules():
            if ("act" in name) and (int(name.split("stages.")[-1].split(".")[0]) >= int(self.pos)):
                new_name = convert_name(name)
                exec("self.model.{}=self.new_act(module)".format(new_name))
    
    def _prep_vit(self):
        for name, module in self.model.named_modules():
            if ("act" in name) and (int(name.split("blocks.")[-1].split(".")[0]) >= int(self.pos)):
                new_name = convert_name(name)
                exec("self.model.{}=self.new_act(module)".format(new_name))
    
    def _prep_swin(self):
        for name, module in self.model.named_modules():
            if ("act" in name) and (int(name.split("layers.")[-1].split(".")[0]) >= int(self.pos)):
                new_name = convert_name(name)
                exec("self.model.{}=self.new_act(module)".format(new_name))
                
    def _prep_effnetv2(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.ReLU, nn.SiLU)):
                if "blocks" in name and int(name.split(".")[3])>=int(self.pos):
                    new_name = convert_name(name)
                    exec("self.model.{}=self.new_act(module)".format(new_name))
    
    def _prep_inceptionv3(self):
        from timm.models.inception_v3 import BasicConv2d
        for name, module in self.model.named_modules():
            if isinstance(module, BasicConv2d):
                new_name = convert_name(name)
                exec("self.model.{}=BasicConv2dNew(module)".format(new_name))
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                new_name = convert_name(name)
                if "Conv2d" in name.split(".")[2]:
                    if int(name.split(".")[2].split("_")[1][0]) >= int(self.pos):
                        exec("self.model.{}=self.new_act(module)".format(new_name))
                if "Mixed" in name.split(".")[2]:
                    if int(name.split(".")[2].split("_")[1][0]) >= int(self.pos):
                        if "pool" not in name.split(".")[3] and "_" in name.split(".")[3]:
                            exec("self.model.{}=self.new_act(module)".format(new_name))
                            
                            
class LinBPNewBackend(NewBackend, LinBP):
    def __init__(self, args, **kwargs):
        NewBackend.__init__(self, args, **kwargs)
        LinBP.__init__(self, args, **kwargs)
        