import os
import shutil
import logging

from .utils.helper import zip_folder

class Uploader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = logging.getLogger("lazyobjectplacement")

        self.data_save_path = self.cfg.openimages.data_save_path
        self.images_folder_path = os.path.join(self.data_save_path, self.cfg.openimages.images_folder_name)
        self.dataset_folder_path = os.path.dirname(self.images_folder_path)

    def run(self, set_name, zip_id):
        zip_folder_path = os.path.join(self.data_save_path, f"{set_name}_{zip_id}.zip")
        zip_folder(self.dataset_folder_path, zip_folder_path)
        shutil.rmtree(self.dataset_folder_path)

        # TODO
        # Upload to cloud storage logic goes here