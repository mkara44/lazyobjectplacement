import os
import shutil
import logging

from .utils.helper import zip_folder, upload_to_gcs

class Uploader:
    def __init__(self, cfg, upload_to_gcs):
        self.cfg = cfg
        self.upload_to_gcs = upload_to_gcs
        self.logger = logging.getLogger("lazyobjectplacement")

        self.data_save_path = self.cfg.openimages.data_save_path
        self.images_folder_path = os.path.join(self.data_save_path, self.cfg.openimages.images_folder_name)
        self.dataset_folder_path = os.path.dirname(self.images_folder_path)

    def run(self, set_name, zip_id):
        zip_folder_path = os.path.join(self.data_save_path, f"{set_name}_{zip_id}.zip")
        zip_folder(self.dataset_folder_path, zip_folder_path)
        shutil.rmtree(self.dataset_folder_path)

        if self.upload_to_gcs:
            self.logger.info(f"Uploading {zip_folder_path} to Google Cloud Storage...")
            bucket_name = os.getenv("GCP_BUCKET_NAME")
            upload_to_gcs(zip_folder_path, bucket_name, f"{set_name}_{zip_id}.zip")