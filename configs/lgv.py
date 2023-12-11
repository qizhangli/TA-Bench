from configs.base import get_config as default_config

def get_config():
    cfg = default_config()
        
    cfg.method = "LGV"
    cfg.model_path = "checkpoints/LGV/lgv_checkpoints"
    cfg.n_models = 1
    
    return cfg
