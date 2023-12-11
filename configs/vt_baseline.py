from configs.base import get_config as default_config

def get_config():
    cfg = default_config()
    
    cfg.method = "VTBaseline"
    
    cfg.sample_times = 20
    cfg.beta=1.5
    
    return cfg
