from configs.base import get_config as default_config
import sys

def get_config():
    cfg = default_config()
    
    if "--config.model_name" in sys.argv:
        cfg.model_name = sys.argv[sys.argv.index("--config.model_name") + 1]
        
    cfg.method = "TAP"
    if cfg.model_name=="tv_resnet50":
        cfg.il_pos = "1,2"
    elif cfg.model_name=="inception_v3":
        cfg.il_pos = "Conv2d_1a_3x3,Conv2d_2b_3x3,Conv2d_3b_1x1,Conv2d_4a_3x3,Mixed_5d,Mixed_6e,Mixed_7c"
    elif cfg.model_name=="vgg19_bn":
        cfg.il_pos = "6,13,19,26,32,39,45,51"
    elif cfg.model_name=="convnext_base":
        cfg.il_pos = "0,1,2,3"
    elif cfg.model_name in ["vit_base_patch16_224", "deit3_base_patch16_224", "beit_base_patch16_224", "mixer_b16_224"]:
        cfg.il_pos = "1,3,5,7,9,11"
    elif cfg.model_name=="tf_efficientnetv2_m":
        cfg.il_pos = "0,1,2,3,4,5,6"
    elif cfg.model_name=="swin_base_patch4_window7_224":
        cfg.il_pos = "0,1,2,3"
    else:
        raise RuntimeError("check the model name")
    cfg.lam = 0.005
    cfg.eta = 0.01
    cfg.alpha = 0.5
    cfg.ks = 3
    
    return cfg
