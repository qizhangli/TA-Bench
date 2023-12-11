import os
import random

import numpy as np
import torch
from absl import app
from ml_collections import config_flags
from torch.backends import cudnn

from attacker import Attacker
from utils import build_dataset, build_generative_model, build_model

_CONFIG = config_flags.DEFINE_config_file('config')

def main(_):
    # Initialize
    cfg = _CONFIG.value
    print(cfg)
    if cfg.constraint == "linf":
        cfg.epsilon = cfg.epsilon / 255.
        cfg.step_size = cfg.step_size / 255.
    SEED = cfg.seed
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.makedirs(cfg.save_dir, exist_ok=True if cfg.force else False)
    
    # Prepare model and data
    if cfg.generative:
        model, data_config = build_generative_model(cfg.model_name, cfg.model_path, cfg.method)
    else:
        model, data_config = build_model(cfg.model_name)
    dataloader = build_dataset(cfg, data_config)
    
    # Attack
    attacker = Attacker(cfg, source_model=model, dataloader=dataloader)
    attacker.attack(verbose=True)

if __name__ == '__main__':
    app.run(main)
