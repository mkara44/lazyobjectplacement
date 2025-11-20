import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from transformers import AutoImageProcessor

from .perturbation import Perturbation
from .utils.helper import crop_foreground_from_mask


class OpenImagesDataset(Dataset):
    def __init__(self, dataset_path, set_name, target_size, dino_model=None, perturbation_cfg=None, extract_features=True):
        super().__init__()
        self.set_name = set_name
        self.target_size = target_size
        self.dataset_path = dataset_path

        self.image_file_list, self.mask_file_list, self.vae_image_features_file_list, self.vae_image_wmask_features_file_list = self.__load_dataset(extract_features)

        self.perturbation = None
        self.vae_transform = None
        self.dino_processor = None

        if perturbation_cfg is not None:
            self.perturbation = Perturbation(**perturbation_cfg)

        if extract_features:
            self.vae_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        
        if dino_model is not None:
            self.dino_processor = AutoImageProcessor.from_pretrained(dino_model)

    def __load_dataset(self, extract_features):
        images_path = os.path.join(self.dataset_path, "images", self.set_name)
        masks_path = os.path.join(self.dataset_path, "masks", self.set_name)

        #mask_file_list = [os.path.join(masks_path, f) for f in os.listdir(masks_path) if f.endswith('.png') or f.endswith('.jpg')]
        mask_file_list = [
            "tmp_data/data/masks/test/0a73d3ed3923b4f7_m06m11_e342670e.png",
            "tmp_data/data/masks/test/0cae8fea8dacf503_m02p5f1q_419c669e.png",
            "tmp_data/data/masks/test/0e2865a812f2d8cb_m0c9ph5_8dfafeca.png"
        ]
        image_id_list = [os.path.basename(f).split('_')[0] for f in mask_file_list]
        image_file_list = [os.path.join(images_path, f"{image_id}.jpg") for image_id in image_id_list]

        if not extract_features:
            vae_image_features = os.path.join(self.dataset_path, "vae_image_features", self.set_name)
            vae_image_features_file_list = [os.path.join(vae_image_features, f) for f in os.listdir(vae_image_features) if f.endswith('.npy')]

            vae_image_wmask_features = os.path.join(self.dataset_path, "vae_image_wmask_features", self.set_name)
            vae_image_wmask_features_file_list = [os.path.join(vae_image_wmask_features, f) for f in os.listdir(vae_image_wmask_features) if f.endswith('.npy')]
        else:
            vae_image_features_file_list = None
            vae_image_wmask_features_file_list = None

        return image_file_list, mask_file_list, vae_image_features_file_list, vae_image_wmask_features_file_list

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, idx):
        image_path = self.image_file_list[idx]
        mask_path = self.mask_file_list[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.target_size)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        mask[mask > 0] = 1

        foreground_image, foreground_mask = crop_foreground_from_mask(image, mask)

        sample = {}
        if self.perturbation is not None:
            foreground_image, foreground_mask = self.perturbation(foreground_image, foreground_mask)

        if self.vae_transform is not None:
            transformed_image = self.vae_transform(image)
            image_wmask = transformed_image * torch.from_numpy(1 - mask).unsqueeze(0)  # Apply mask
            sample["image"] = transformed_image
            sample["image_wmask"] = image_wmask

        else:
            vae_image_features_path = self.vae_image_features_file_list[idx]
            vae_image_wmask_features_path = self.vae_image_wmask_features_file_list[idx]
            sample["image_features"] = torch.from_numpy(np.load(vae_image_features_path)).squeeze(0)
            sample["image_wmask_features"] = torch.from_numpy(np.load(vae_image_wmask_features_path)).squeeze(0)
    
        if self.dino_processor is not None:
            foreground_image = self.dino_processor(images=foreground_image, return_tensors="pt")["pixel_values"].squeeze(0)

        sample["image_name"] = os.path.basename(image_path)
        sample["mask_name"] = os.path.basename(mask_path)
        sample["mask"] = torch.from_numpy(mask).unsqueeze(0)
        sample["foreground_image"] = foreground_image
        return sample