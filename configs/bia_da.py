from configs.base import get_config as default_config
import sys

def get_config():
    cfg = default_config()
        
    cfg.method = "BIADA"
    cfg.generative = True
    cfg.model_name = "GeneratorResnet"
    cfg.model_path = "checkpoints/BIA_DA/netG_BIA+DA_0.pth"
    
    return cfg
