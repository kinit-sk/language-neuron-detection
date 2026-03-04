import os

import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_ex_id_from_config_name() -> str:
    if not HydraConfig.initialized():
        raise RuntimeError("HydraConfig is not initialized; cannot derive ex_id from config filename")
    config_name = HydraConfig.get().job.config_name
    if not config_name:
        raise RuntimeError("Hydra job config_name is missing; cannot derive ex_id")
    return os.path.splitext(os.path.basename(str(config_name)))[0]


def get_pipeline_step(cfg: DictConfig, step_name: str) -> DictConfig:
    pipeline_cfg = cfg.get("pipeline")
    if pipeline_cfg is None:
        raise KeyError("Missing required top-level config section: pipeline")

    step_cfg = pipeline_cfg.get(step_name)
    if step_cfg is None:
        raise KeyError(f"Missing required pipeline step config: pipeline.{step_name}")
    return step_cfg
