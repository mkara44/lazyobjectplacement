import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from diffusers import AutoencoderKL

from src.train.dataset import OpenImagesDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Extract VAE Features for LazyObjectPlacementModel")

    # Data parameters
    parser.add_argument("--config_path", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--set_name", type=str, required=True, choices=["train", "test", "validation"], help="Dataset set name (e.g., train, test, validation)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")

    return parser.parse_args()

def main(args, cfg):
    vae_image_features_folder_path = os.path.join(args.data_path, cfg.preprocess.openimages.vae_image_features_folder_name, args.set_name)
    vae_image_wmask_features_folder_path = os.path.join(args.data_path, cfg.preprocess.openimages.vae_image_wmask_features_folder_name, args.set_name)
    os.makedirs(vae_image_features_folder_path, exist_ok=True)
    os.makedirs(vae_image_wmask_features_folder_path, exist_ok=True)

    # Initialize models
    vae = AutoencoderKL.from_pretrained(
        cfg.train.vae_pretrained_model, 
    )

    vae.to(args.device)
    vae.requires_grad_(False)

    # Load dataset
    dataset = OpenImagesDataset(
        dataset_path=args.data_path,
        set_name=args.set_name,
        dino_model=cfg.train.object_encoder_model,
        extract_features=True,
        target_size=cfg.train.dataset.target_size
    )

    for sample in tqdm(dataset, desc="Extracting VAE features"):
        with torch.no_grad():
            mask_name = sample["mask_name"]
            image_name = sample["image_name"]
            image = sample["image"].unsqueeze(0).to(args.device)
            image_wmask = sample["image_wmask"].unsqueeze(0).to(args.device)

            vae_image_features_file_path = os.path.join(vae_image_features_folder_path, f"{os.path.splitext(image_name)[0]}.npy")
            vae_image_wmask_features_file_path = os.path.join(vae_image_wmask_features_folder_path, f"{os.path.splitext(mask_name)[0]}.npy")
            
            if not os.path.exists(vae_image_features_file_path):
                vae_image_latents = vae.encode(image).latent_dist.sample()
                vae_image_latents = vae_image_latents * vae.config.scaling_factor
                vae_image_latents = vae_image_latents.cpu().numpy()
                np.save(vae_image_features_file_path, vae_image_latents)

            vae_image_wmask_latents = vae.encode(image_wmask).latent_dist.sample()
            vae_image_wmask_latents = vae_image_wmask_latents * vae.config.scaling_factor
            vae_image_wmask_latents = vae_image_wmask_latents.cpu().numpy()
            np.save(vae_image_wmask_features_file_path, vae_image_wmask_latents)

if __name__ == "__main__":
    args = parse_args()
    cfg = OmegaConf.load(args.config_path)

    main(args, cfg)