from configs.base import get_config as default_config

def get_config():
    cfg = default_config()
    
    cfg.method = "IRBaseline"
    
    cfg.sample_grid_num = 128
    cfg.grid_scale=16
    cfg.sample_times=65
    
    return cfg
