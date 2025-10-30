import logging
import argparse
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.utils.helper import setup_logger
from src.preprocess.downloader import Downloader
from src.preprocess.mask_controller import MaskController
from src.preprocess.uploader import Uploader

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess OpenImages Dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--upload_to_gcs",
        action="store_true",
        help="Flag to upload processed data to Google Cloud Storage.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = OmegaConf.load(args.config)
    setup_logger(cfg.logging)
    logger = logging.getLogger("lazyobjectplacement")

    # Only segmentation masks are handled in this script
    # Bounding boxes are very noisy and will not be used in the current version
    downloader_obj = Downloader(cfg.preprocess)
    mask_controller_obj = MaskController(cfg.preprocess)
    uploader_obj = Uploader(cfg.preprocess, args.upload_to_gcs)
    for set_name in ['train', 'validation', 'test']:
        downloader_obj.download_base_files(set_name)

        for zip_id in cfg.preprocess.openimages.id_list:
            logger.info(f"Downloading files for {set_name} with zip ID {zip_id}...")
            downloader_obj.download_files(set_name, zip_id)
            
            logger.info(f"Controlling masks for {set_name} with zip ID {zip_id}...")
            mask_controller_obj.run(set_name, zip_id)

            logger.info(f"Uploading files for {set_name} with zip ID {zip_id}...")
            uploader_obj.run(set_name, zip_id)

    logger.info("Preprocessing completed successfully.")
    