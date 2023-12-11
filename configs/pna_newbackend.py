from configs.newbackend import get_config as default_config

def get_config():
    cfg = default_config()
        
    cfg.method = "PNANewBackend"
    
    return cfg
