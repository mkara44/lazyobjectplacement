from omegaconf import OmegaConf

from src.utils.helper import setup_logger
from src.preprocess.downloader import Downloader

if __name__ == "__main__":
    cfg = OmegaConf.load("config.yaml")
    setup_logger(cfg.logging)

    dp = Downloader(cfg.preprocess)
    dp.run()
