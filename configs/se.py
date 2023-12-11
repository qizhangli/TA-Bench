from configs.base import get_config as default_config
import sys

def get_config():
    cfg = default_config()
    
    if "--config.model_name" in sys.argv:
        cfg.model_name = sys.argv[sys.argv.index("--config.model_name") + 1]
        
    cfg.method = "SE"
    if cfg.model_name in ["vit_base_patch16_224", "deit3_base_patch16_224", "beit_base_patch16_224", "mixer_b16_224"]:
        cfg.il_pos = "0,1,2,3,4,5,6,7,8,9,11"
    else:
        raise RuntimeError("check the model name")
    
    return cfg
