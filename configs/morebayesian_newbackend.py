from configs.newbackend import get_config as default_config

def get_config():
    cfg = default_config()
        
    cfg.method = "MoreBayesianNewBackend"
    cfg.model_path = "checkpoints/MoreBayesian/morebayesian.pt"
    cfg.morebayesian_scale = 1
    cfg.n_models = 1
    
    return cfg
