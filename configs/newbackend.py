from configs.base import get_config as default_config

def get_config():
    cfg = default_config()
    
    cfg.method = "NewBackend"
    
    cfg.di_resize_rate = 0.9
    cfg.di_diversity_prob = 0.5
    cfg.ti_len = 3
    cfg.ps_npatch = 128
    cfg.mi_mu, cfg.ni_mu = 1, 1
    
    return cfg
