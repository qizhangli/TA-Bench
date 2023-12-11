from configs.base import get_config as default_config

def get_config():
    cfg = default_config()
        
    cfg.method = "CDA"
    cfg.generative = True
    cfg.model_name = "GeneratorResnet"
    cfg.model_path = "checkpoints/CDA/netG_-1_img_res152_imagenet_0_rl.pth"
    
    return cfg
