from configs.base import get_config as default_config

def get_config():
    cfg = default_config()
        
    cfg.method = "DRA"
    cfg.model_path = 'checkpoints/DRA/DRA_resnet50.pth'
    
    return cfg
