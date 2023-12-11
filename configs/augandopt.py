from configs.base import get_config as default_config
import sys

def get_config():
    cfg = default_config()
    
    cfg.method = "AugandOpt"
    
    if "--config.combine" in sys.argv:
        cfg.combine = sys.argv[sys.argv.index("--config.combine") + 1]
    else:
        raise RuntimeError("please set --config.combine")
    
    if "DI" in cfg.combine:
        cfg.di_resize_rate = 0.9
        cfg.di_diversity_prob = 0.5
    else:
        cfg.di_resize_rate, cfg.di_diversity_prob = 1., 0
    if "SI" in cfg.combine:
        cfg.si_m = 5
    else:
        cfg.si_m = 1
    if "Admix" in cfg.combine:
        cfg.si_m, cfg.admix_m, cfg.admix_eta = 5, 1, 0.2
    elif "SI" in cfg.combine:
        cfg.si_m, cfg.admix_m, cfg.admix_eta = 5, 1, 0
    else:
        cfg.si_m, cfg.admix_m, cfg.admix_eta = 1, 1, 0
    if "TIori" in cfg.combine:
        cfg.ti_kernel = "gkern_7_3"
        cfg.ti_len = 0
    elif "TI" in cfg.combine:
        cfg.ti_len = 3
        cfg.ti_kernel = None
    else:
        cfg.ti_kernel = None
        cfg.ti_len = 0
    cfg.ps, cfg.ps_npatch = False, None
    if "DP" in cfg.combine:
        cfg.ps = True
        cfg.ps_npatch = 128
    cfg.pre_grad = False
    if "NI" in cfg.combine:
        cfg.mi_mu, cfg.ni_mu = 1, 1
    elif "PI" in cfg.combine:
        cfg.mi_mu, cfg.ni_mu = 1, 1
        cfg.pre_grad = True
    elif "MI" in cfg.combine:
        cfg.mi_mu, cfg.ni_mu = 1, 0
    else:
        cfg.mi_mu, cfg.ni_mu = 0, 0
    cfg.UN = True if "UN" in cfg.combine else False
    cfg.pgd = True if "PGD" in cfg.combine else False
    
    return cfg
