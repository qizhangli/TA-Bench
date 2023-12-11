from configs.base import get_config as default_config

def get_config():
    cfg = default_config()
    
    cfg.method = "TAIGBaseline"
    
    cfg.sample_times = 30
    
    
    return cfg
