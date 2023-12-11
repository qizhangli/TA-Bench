from configs.newbackend import get_config as default_config
import sys

def get_config():
    cfg = default_config()
    
    if "--config.model_name" in sys.argv:
        cfg.model_name = sys.argv[sys.argv.index("--config.model_name") + 1]
        
    cfg.method = "ConBPNewBackend"
    if cfg.model_name=="tv_resnet50":
        cfg.pos = "4"
    elif cfg.model_name=="inception_v3":
        cfg.pos = "7"
    elif cfg.model_name=="vgg19_bn":
        cfg.pos = "52"
    else:
        raise RuntimeError("check the model name")
    
    
    return cfg
