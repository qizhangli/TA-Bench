from configs.newbackend import get_config as default_config
import sys

def get_config():
    cfg = default_config()
    
    if "--config.model_name" in sys.argv:
        cfg.model_name = sys.argv[sys.argv.index("--config.model_name") + 1]
        
    cfg.method = "ILAppNewBackend"
    if cfg.model_name=="tv_resnet50":
        cfg.il_pos = "4"
    elif cfg.model_name=="inception_v3":
        cfg.il_pos = "Mixed_6e"
    elif cfg.model_name=="vgg19_bn":
        cfg.il_pos = "45"
    elif cfg.model_name=="convnext_base":
        cfg.il_pos = "3"
    elif cfg.model_name == "vit_base_patch16_224":
        cfg.il_pos = "11"
    elif cfg.model_name == "deit3_base_patch16_224":
        cfg.il_pos = "9"
    elif cfg.model_name == "beit_base_patch16_224":
        cfg.il_pos = "11"
    elif cfg.model_name == "mixer_b16_224":
        cfg.il_pos = "9"
    elif cfg.model_name=="tf_efficientnetv2_m":
        cfg.il_pos = "6"
    elif cfg.model_name=="swin_base_patch4_window7_224":
        cfg.il_pos = "3"
    else:
        raise RuntimeError("check the model name")
    
    cfg.base_steps=101
    cfg.njobs=10
    cfg.rand_restart=0
    cfg.linr="svr"
    
    return cfg
