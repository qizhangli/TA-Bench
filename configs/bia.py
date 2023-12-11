from configs.base import get_config as default_config

def get_config():
    cfg = default_config()
        
    cfg.method = "BIA"
    cfg.generative = True
    cfg.model_name = "GeneratorResnet"
    cfg.model_path = "checkpoints/BIA/netG_BIA_0.pth"
    
    return cfg
