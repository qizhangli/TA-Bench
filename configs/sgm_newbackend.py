from configs.newbackend import get_config as default_config
import sys

def get_config():
    cfg = default_config()
    
    if "--config.model_name" in sys.argv:
        cfg.model_name = sys.argv[sys.argv.index("--config.model_name") + 1]
        
    cfg.method = "SGMNewBackend"
    if cfg.model_name=="tv_resnet50":
        cfg.sgm_gamma = 0.7
    elif cfg.model_name=="convnext_base":
        cfg.sgm_gamma = 0.6
    elif cfg.model_name == "vit_base_patch16_224":
        cfg.sgm_gamma = 0.8
    elif cfg.model_name == "deit3_base_patch16_224":
        cfg.sgm_gamma = 0.6
    elif cfg.model_name == "beit_base_patch16_224":
        cfg.sgm_gamma = 0.7
    elif cfg.model_name == "mixer_b16_224":
        cfg.sgm_gamma = 0.7
    elif cfg.model_name=="tf_efficientnetv2_m":
        cfg.sgm_gamma = 0.5
    elif cfg.model_name=="swin_base_patch4_window7_224":
        cfg.sgm_gamma = 0.6
    else:
        raise RuntimeError("check the model name")
    
    return cfg
