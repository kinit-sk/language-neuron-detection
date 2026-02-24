import os

from hydra.core.hydra_config import HydraConfig
import torch


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device



def set_ex_id_from_config_name() -> str:
    if not HydraConfig.initialized():
        raise RuntimeError("HydraConfig is not initialized; cannot derive ex_id from config filename")
    config_name = HydraConfig.get().job.config_name
    if not config_name:
        raise RuntimeError("Hydra job config_name is missing; cannot derive ex_id")
    return os.path.splitext(os.path.basename(str(config_name)))[0]
