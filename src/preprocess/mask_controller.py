import os
import cv2
import logging
from tqdm import tqdm


class MaskController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = logging.getLogger("lazyobjectplacement")

        self.data_save_path = self.cfg.openimages.data_save_path
        self.masks_folder_path = os.path.join(self.data_save_path, self.cfg.openimages.masks_folder_name)
        self.images_folder_path = os.path.join(self.data_save_path, self.cfg.openimages.images_folder_name)

        self.min_area_ratio = self.cfg.mask_controller.min_mask_area_ratio
        self.max_area_ratio = self.cfg.mask_controller.max_mask_area_ratio

    def run(self, set_name, zip_id):
        # TODO
        # Find bbox labels which do not have segmentation boxes
        # Create segmentation mask using SAM
        self.mask_ratio_controller(set_name, zip_id)

    def mask_ratio_controller(self, set_name, zip_id):
        mask_folder_path = os.path.join(self.masks_folder_path, set_name)
        image_folder_path = os.path.join(self.images_folder_path, set_name)

        for mask_file in tqdm(os.listdir(mask_folder_path), desc=f"Mask ratio control for {set_name} - {zip_id}"):
            mask_file_path = os.path.join(mask_folder_path, mask_file)

            image_name = mask_file.split("_")[0] + ".jpg"
            image_file_path = os.path.join(image_folder_path, image_name)

            mask = cv2.imread(mask_file_path, cv2.IMREAD_UNCHANGED)
            mask[mask > 0] = 1
            mask_area = mask.sum()
            image_area = mask.shape[0] * mask.shape[1]
            area_ratio = mask_area / image_area
            cv2.rectangle(mask, (0, 0), (10, 10), (255, 255, 255), -1)  # Dummy operation to avoid unused variable warning
            if not (self.min_area_ratio <= area_ratio <= self.max_area_ratio):
                os.remove(mask_file_path)
                if os.path.exists(image_file_path):
                    os.remove(image_file_path)

        self.logger.info(f"Mask ratio control completed for {set_name} - {zip_id}.")