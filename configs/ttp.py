from configs.base import get_config as default_config

def get_config():
    cfg = default_config()
        
    cfg.method = "TTP"
    cfg.generative = True
    cfg.model_name = "GeneratorResnet"
    cfg.model_path = ["checkpoints/TTP/netG_res152_IN_19_24.pth",
                      "checkpoints/TTP/netG_res152_IN_19_919.pth"]
    cfg.ttp_target = [24,919]
    
    return cfg
