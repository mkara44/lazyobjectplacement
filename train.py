import torch
import argparse
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler

from src.models.context_encoder import GlobalContextEncoder
from src.models.pixart_decoder import PixArtDecoder
from src.models.utils.helper import Dinov2FeatureExtractor, get_hole


def parse_args():
    parser = argparse.ArgumentParser(description="Train LazyObjectPlacementModel")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (e.g., 'cuda' or 'cpu')")
    
    # Data parameters
    #parser.add_argument("--data_path", type=str, required=True, help="Path to training data")

    # Model parameters
    parser.add_argument("--vae_pretrained_model", type=str, default="madebyollin/sdxl-vae-fp16-fix", help="Pretrained VAE model name or path")
    parser.add_argument("--object_encoder_model", type=str, default="facebook/dinov2-small", choices=["facebook/dinov2-small"], help="Pretrained object encoder model name or path")

    return parser.parse_args()

import random
def generate_mask():
    mask = torch.zeros((1, 1, 512, 512))
    mask[:, :, random.randint(0, 20):random.randint(20, 400), random.randint(0, 20):random.randint(20, 400)] = 1.0
    #mask[:, :, :, :] = 1.0
    return mask

def main(args):
    # Initialize models
    vae = AutoencoderKL.from_pretrained(
        args.vae_pretrained_model, 
    )

    if args.object_encoder_model == "facebook/dinov2-small":
        object_encoder = Dinov2FeatureExtractor()

    pixart_decoder = PixArtDecoder()
    global_context_encoder = GlobalContextEncoder()

    vae.to(args.device)
    vae.requires_grad_(False)

    object_encoder.to(args.device)
    global_context_encoder.to(args.device)
    pixart_decoder.to(args.device)

    noise_scheduler = DDPMScheduler.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-512x512",
        subfolder="scheduler"
    )


    tmp_dataset = {
        "image": torch.randn(args.batch_size, 3, 512, 512),
        "mask": generate_mask().repeat(1, 1, 1, 1),
        #"mask": torch.cat([generate_mask().repeat(1, 1, 1, 1) for _ in range(args.batch_size)], dim=0),
        "object_image": torch.randn(args.batch_size, 3, 512, 512),
    }

    for epoch in range(args.num_epochs):
        train_loss = 0.0
        for step, batch in enumerate([tmp_dataset]):
            # prepare latents
            image = batch["image"].to(args.device)
            mask = batch["mask"].to(args.device)
            object_image = batch["object_image"].to(args.device)

            with torch.no_grad():
                Z = vae.encode(image*(1-mask)).latent_dist.sample()
                Z = Z * vae.config.scaling_factor

                X = vae.encode(image).latent_dist.sample()
                X = X * vae.config.scaling_factor

            if Z.shape[2:] != mask.shape[2:]:
                mask = F.interpolate(
                    mask,
                    size=Z.shape[2:],
                    mode='nearest'
                )

            T_all = global_context_encoder(Z, mask)
            T_hole, _ = get_hole(T_all, mask, global_context_encoder.num_patches)

            X_all, height, width = pixart_decoder.patchify(X)
            X_hole, bool_mask = get_hole(X_all, mask, 1024)
            
            object_features = object_encoder(object_image)

            timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (args.batch_size,), device=args.device
                )

            model_pred = pixart_decoder(
                x_hole = X_hole,
                t_hole = T_hole,
                object_features = object_features,
                timestep = timesteps,
                height = height,
                width = width,
                bool_mask = bool_mask,
            )

            print(model_pred.shape)
            
    print("DONE!")

if __name__ == "__main__":
    args = parse_args()
    main(args)