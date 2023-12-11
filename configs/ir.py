from configs.base import get_config as default_config

def get_config():
    cfg = default_config()
    
    cfg.method = "IR"
    
    cfg.sample_grid_num = 32
    cfg.grid_scale=16
    cfg.sample_times=32
    cfg.lam=1
    
    return cfg
