import torch

# Allow Multi-GPUs
class ILOutHook(object):
    def __init__(self, model_name, il_pos):
        self.model_name = model_name
        self.il_pos = il_pos
        self.pos_ls = self.il_pos.split(",")
        self.multi_ilout_dict = {}
        
    def _get_hook_ilout(self, cache_dict):
        def _hook_ilout(module, input, output):
            cache_dict[input[0].device.index] = output
        return _hook_ilout
    
    def _get_ilout(self, clear=True):
        ilout_ls = []
        for pos in self.pos_ls:
            ilout_ls.append(
                self._get_ilout_once(self.multi_ilout_dict[str(pos)], clear))
        return ilout_ls if len(ilout_ls)>1 else ilout_ls[0]
            
    def _get_ilout_once(self, cache_dict, clear=True):
        ilout = []
        for i in range(torch.cuda.device_count()):
            ilout.append(cache_dict[i].clone().to("cuda:0"))
            if clear:
                cache_dict[i] = None
        ilout = torch.cat(ilout, dim = 0)
        return ilout
    
    def _get_ilout_grad(self, loss, clear=True):
        ilout_grad_ls = []
        for i, pos in enumerate(self.pos_ls):
            ilout_grad_ls.append(
                self._get_ilout_grad_once(self.multi_ilout_dict[str(pos)], 
                                          loss, (i+1 == len(self.pos_ls)), clear))
        return ilout_grad_ls if len(ilout_grad_ls)>1 else ilout_grad_ls[0]
    
    def _get_ilout_grad_once(self, cache_dict, loss, last=False, clear=True):
        ilout_grad = []
        for i in range(torch.cuda.device_count()):
            grad = torch.autograd.grad(loss, cache_dict[i], retain_graph=(not last or i+1 < torch.cuda.device_count()))[0].data.clone().to("cuda:0")
            ilout_grad.append(grad)
            if clear:
                cache_dict[i] = None
        ilout_grad = torch.cat(ilout_grad, dim = 0)
        return ilout_grad
    
    def _prep_hook(self):
        for pos in self.pos_ls:
            if self.model_name in ["tv_resnet50"]:
                self.il_module = eval("self.model.module[1].layer{}".format(pos))
            elif self.model_name in ["inception_v3"]:
                self.il_module = eval("self.model.module[1].{}".format(pos))
            elif self.model_name in ["vgg19_bn"]:
                self.il_module = eval("self.model.module[1].features[{}]".format(pos))
            elif self.model_name in ["convnext_base"]:
                self.il_module = eval("self.model.module[1].stages[{}]".format(pos))
            elif self.model_name in ["vit_base_patch16_224", "deit3_base_patch16_224", "beit_base_patch16_224", "mixer_b16_224", "tf_efficientnetv2_m"]:
                self.il_module = eval("self.model.module[1].blocks[{}]".format(pos))
            elif self.model_name in ["swin_base_patch4_window7_224"]:
                self.il_module = eval("self.model.module[1].layers[{}]".format(pos))
            else:
                raise RuntimeError("not support")
            self.multi_ilout_dict[str(pos)] = {}
            self.il_module.register_forward_hook(self._get_hook_ilout(self.multi_ilout_dict[str(pos)]))

