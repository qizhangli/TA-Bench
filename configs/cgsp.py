from configs.base import get_config as default_config

def get_config():
    cfg = default_config()
        
    cfg.method = "CGSP"
    cfg.generative = True
    cfg.model_name = "ConGeneratorResnet"
    cfg.model_path = "checkpoints/CGSP/model-res152-epoch9.pth"
    
    return cfg
