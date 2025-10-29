import logging
from omegaconf import OmegaConf

from src.utils.helper import setup_logger
from src.preprocess.downloader import Downloader
from src.preprocess.mask_controller import MaskController
from src.preprocess.uploader import Uploader


if __name__ == "__main__":
    cfg = OmegaConf.load("config.yaml")
    setup_logger(cfg.logging)
    logger = logging.getLogger("lazyobjectplacement")

    downloader_obj = Downloader(cfg.preprocess)
    mask_controller_obj = MaskController(cfg.preprocess)
    uploader_obj = Uploader(cfg.preprocess)
    for set_name in ["test"]: #['train', 'validation', 'test']:
        downloader_obj.download_base_files(set_name)

        for zip_id in cfg.preprocess.openimages.id_list:
            #logger.info(f"Downloading files for {set_name} with zip ID {zip_id}...")
            downloader_obj.download_files(set_name, zip_id)
            
            #logger.info(f"Controlling masks for {set_name} with zip ID {zip_id}...")
            #mask_controller_obj.run(set_name, zip_id)

            #logger.info(f"Uploading files for {set_name} with zip ID {zip_id}...")
            #uploader_obj.run(set_name, zip_id)
            
            break

    logger.info("Preprocessing completed successfully.")
    