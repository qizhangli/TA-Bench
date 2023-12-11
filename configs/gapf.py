from configs.base import get_config as default_config

def get_config():
    cfg = default_config()
        
    cfg.method = "GAPF"
    cfg.generative = True
    cfg.model_name = "GeneratorResnet"
    cfg.model_path = "checkpoints/GAPF/1_net_G.pth"
    
    return cfg
