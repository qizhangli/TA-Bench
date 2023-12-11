from configs.base import get_config as default_config

def get_config():
    cfg = default_config()
        
    cfg.method = "RFA"
    cfg.model_path = 'checkpoints/RFA/resnet50_linf_eps0.5.ckpt'
    
    return cfg
