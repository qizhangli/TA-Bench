from configs.base import get_config as default_config

def get_config():
    cfg = default_config()
        
    cfg.method = "BIARN"
    cfg.generative = True
    cfg.model_name = "GeneratorResnet"
    cfg.model_path = "checkpoints/BIA_RN/netG_BIA+RN_0.pth"
    
    return cfg
