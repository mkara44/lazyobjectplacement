from omegaconf import OmegaConf
import logging.config


def setup_logger(logging_cfg):
    logging_cfg = OmegaConf.to_container(logging_cfg, resolve=True)
    logging.config.dictConfig(logging_cfg)
