from configs.newbackend import get_config as default_config
import sys

def get_config():
    cfg = default_config()
    
    if "--config.model_name" in sys.argv:
        cfg.model_name = sys.argv[sys.argv.index("--config.model_name") + 1]
        
    cfg.method = "LinBPNewBackend"
    if cfg.model_name=="tv_resnet50":
        cfg.pos = "4"
    elif cfg.model_name=="inception_v3":
        cfg.pos = "7"
    elif cfg.model_name=="vgg19_bn":
        cfg.pos = "52"
    elif cfg.model_name=="convnext_base":
        cfg.pos = "3"
    elif cfg.model_name == "vit_base_patch16_224":
        cfg.pos = "10"
    elif cfg.model_name == "deit3_base_patch16_224":
        cfg.pos = "10"
    elif cfg.model_name == "beit_base_patch16_224":
        cfg.pos = "10"
    elif cfg.model_name == "mixer_b16_224":
        cfg.pos = "10"
    elif cfg.model_name=="tf_efficientnetv2_m":
        cfg.pos = "5"
    elif cfg.model_name=="swin_base_patch4_window7_224":
        cfg.pos = "3"
    else:
        raise RuntimeError("check the model name")
    
    
    return cfg
