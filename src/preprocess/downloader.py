import os
import logging

from .utils.original_downloader import download_all_images
from .utils.helper import download_file, unzip_file, merge_folders, parse_image_id


class Downloader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = logging.getLogger("lazyobjectplacement")

        self.data_save_path = self.cfg.openimages.data_save_path
        self.base_file_folder_path = os.path.join(self.data_save_path, self.cfg.openimages.base_file_folder_name)
        self.images_folder_path = os.path.join(self.data_save_path, self.cfg.openimages.images_folder_name)
        self.masks_folder_path = os.path.join(self.data_save_path, self.cfg.openimages.masks_folder_name)

        os.makedirs(self.data_save_path, exist_ok=True)
        os.makedirs(self.base_file_folder_path, exist_ok=True)
        os.makedirs(self.images_folder_path, exist_ok=True)
        os.makedirs(self.masks_folder_path, exist_ok=True)

    def download_files(self, set_name, zip_id):
        self.download_segmentation_masks(set_name, zip_id)
        self.download_images(set_name, zip_id)

    def download_base_files(self, set_name):
        image_id_url = self.cfg.openimages.image_id_base_url.replace("<SET_NAME>", set_name)
        image_id_target_path = os.path.join(self.base_file_folder_path, f"{set_name}_filelist.csv")
        download_file(image_id_url, target_path=image_id_target_path)

        mask_data_url = self.cfg.openimages.mask_data_url.replace("<SET_NAME>", set_name)
        mask_data_target_path = os.path.join(self.base_file_folder_path, f"{set_name}_mask_data.csv")
        download_file(mask_data_url, target_path=mask_data_target_path)

        bounding_boxes_data_url = self.cfg.openimages.bounding_boxes_base_url.replace("<SET_NAME>", set_name)
        bounding_boxes_target_path = os.path.join(self.base_file_folder_path, f"{set_name}_bbox_annotations.csv")
        download_file(bounding_boxes_data_url, target_path=bounding_boxes_target_path)

        self.logger.info(f"Base file lists downloaded for {set_name}.")

    def download_segmentation_masks(self, set_name, zip_id):
        mask_set_path = os.path.join(self.masks_folder_path, set_name)
        os.makedirs(mask_set_path, exist_ok=True)

        mask_url = self.cfg.openimages.segmentation_masks_base_url.replace("<SET_NAME>", set_name)
        mask_url = mask_url.replace("<ID>", zip_id)

        mask_target_path = os.path.join(mask_set_path, f"{zip_id}.zip")
        download_file(mask_url, target_path=mask_target_path)
        unzip_file(mask_target_path, remove_zip=True)
        merge_folders(mask_set_path)

        self.logger.info(f"All segmentation masks for {set_name} with zip ID {zip_id} downloaded.")

    def download_images(self, set_name, zip_id):
        mask_set_path = os.path.join(self.masks_folder_path, set_name)
        image_set_path = os.path.join(self.images_folder_path, set_name)
        os.makedirs(image_set_path, exist_ok=True)
        
        image_id_list = parse_image_id(mask_set_path, set_name)
        download_all_images(image_id_list, image_set_path, num_processes=self.cfg.openimages.num_processes)
        self.logger.info(f"All images for {set_name} with zip ID{zip_id} downloaded.")