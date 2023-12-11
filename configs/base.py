from ml_collections import config_dict

def get_config():
    cfg = config_dict.ConfigDict()
    cfg.data_dir = "/dockerdata/val/"
    cfg.data_info_dir = "data/rand_5000.csv"
    cfg.save_dir="adv_imgs"
    
    cfg.constraint="linf" # linf / l2
    cfg.epsilon=8.0
    cfg.step_size=1.0
    cfg.model_name="tv_resnet50"
    cfg.steps=100
    cfg.aug_times=1
    
    cfg.batch_size=100
    cfg.force=True
    cfg.seed=0
    cfg.num_workers=4
    
    cfg.generative=False
    cfg.combine="-"
    
    return cfg
