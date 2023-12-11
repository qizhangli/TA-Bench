from configs.base import get_config as default_config

def get_config():
    cfg = default_config()
    
    cfg.method = "IFGSM"
    
    return cfg
