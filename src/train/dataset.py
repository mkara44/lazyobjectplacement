import os
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from transformers import AutoImageProcessor

from .perturbation import Perturbation
from .helper import crop_foreground_from_mask


class OpenImagesDataset(Dataset):
    def __init__(self, dataset_path, set_name, dino_model, target_size, perturbation_cfg):
        super().__init__()
        self.set_name = set_name
        self.target_size = target_size
        self.dataset_path = dataset_path

        self.image_file_list, self.mask_file_list = self.__load_dataset()

        self.perturbation = Perturbation(**perturbation_cfg)
        self.vae_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_model)

    def __load_dataset(self):
        images_path = os.path.join(self.dataset_path, "images", self.set_name)
        masks_path = os.path.join(self.dataset_path, "masks", self.set_name)

        mask_file_list = [os.path.join(masks_path, f) for f in os.listdir(masks_path) if f.endswith('.png') or f.endswith('.jpg')]
        image_id_list = [os.path.basename(f).split('_')[0] for f in mask_file_list]
        image_file_list = [os.path.join(images_path, f"{image_id}.jpg") for image_id in image_id_list]
        return image_file_list, mask_file_list

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, idx):
        image = self.image_file_list[idx]
        mask = self.mask_file_list[idx]

        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.target_size)

        mask = cv2.imread(mask, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        mask[mask > 0] = 1

        foreground_image, foreground_mask = crop_foreground_from_mask(image, mask)
        foreground_image, foreground_mask = self.perturbation(foreground_image, foreground_mask)

        transformed_image = self.vae_transform(image)
        image_wmask = self.vae_transform(image) * (1 - torch.from_numpy(mask).unsqueeze(0))  # Apply mask
        foreground_image = self.dino_processor(images=foreground_image, return_tensors="pt")["pixel_values"].squeeze(0)

        sample = {
            "image": transformed_image,
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "image_wmask": image_wmask,
            "foreground_image": foreground_image
        }
        return sample