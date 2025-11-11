import os
import torch
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL

from src.models.context_encoder import GlobalContextEncoder
from src.models.pixart_decoder import PixArtDecoder
from src.models.utils.helper import Dinov2FeatureExtractor, get_hole
from src.train.dataset import OpenImagesDataset
from src.train.utils.helper import save_checkpoint
from src.utils.iddpm import IDDPM


def parse_args():
    parser = argparse.ArgumentParser(description="Train LazyObjectPlacementModel")

    # Data parameters
    parser.add_argument("--config_path", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")

    return parser.parse_args()

def main(args, cfg):
    accelerator = Accelerator(
            mixed_precision=cfg.mixed_precision,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            project_dir=os.path.join(cfg.work_dir, "logs")
        )

    # Initialize models
    vae = AutoencoderKL.from_pretrained(
        cfg.vae_pretrained_model, 
    )

    if cfg.object_encoder_model == "facebook/dinov2-small":
        object_encoder = Dinov2FeatureExtractor()

    pixart_decoder = PixArtDecoder(pixart_pretrained_model=cfg.diffusers_model)
    global_context_encoder = GlobalContextEncoder()

    vae.to(args.device)
    object_encoder.to(args.device)
    global_context_encoder.to(args.device)
    pixart_decoder.to(args.device)
    
    vae.requires_grad_(False)
    object_encoder.requires_grad_(False)
    global_context_encoder.train()
    pixart_decoder.train()

    train_diffusion = IDDPM(
        str(cfg.train_sampling_steps),
        learn_sigma=cfg.learn_sigma,
        pred_sigma=cfg.pred_sigma,
        snr=cfg.snr_loss
    )

    # Initialize optimizer
    params_to_optimize = list(global_context_encoder.parameters()) + list(pixart_decoder.parameters())
    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=cfg.learning_rate,
                                  betas=(cfg.adam_beta1, cfg.adam_beta2),
                                  weight_decay=cfg.weight_decay,
                                  eps=cfg.adam_epsilon)

    global_context_encoder, pixart_decoder = accelerator.prepare(
        global_context_encoder, pixart_decoder
    )

    # Load dataset
    train_dataset = OpenImagesDataset(
        dataset_path=args.data_path,
        set_name="train",
        dino_model=cfg.object_encoder_model,
        **cfg.dataset
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    optimizer, train_dataloader = accelerator.prepare(optimizer, train_dataloader)

    progress_bar = tqdm(
        range(0, cfg.max_train_steps),
        initial=0,
        desc="Steps"
    )

    global_step = 0
    for epoch in range(1, cfg.num_epochs + 1):
        for step, batch in enumerate(train_dataloader):
            # prepare latents
            image = batch["image"].to(accelerator.device)
            mask = batch["mask"].to(accelerator.device)
            image_wmask = batch["image_wmask"].to(accelerator.device)
            foreground_image = batch["foreground_image"].to(accelerator.device)

            with torch.no_grad():
                Z = vae.encode(image_wmask).latent_dist.sample()
                Z = Z * vae.config.scaling_factor

                X = vae.encode(image).latent_dist.sample()
                X = X * vae.config.scaling_factor

                object_features = object_encoder(foreground_image)

            # Original network uses LearnedMaskDownsampler, here we use nearest neighbor downsampling for simplicity
            if Z.shape[2:] != mask.shape[2:]:
                mask = F.interpolate(
                    mask,
                    size=Z.shape[2:],
                    mode='nearest'
                )

            timestep = torch.randint(
                    0, cfg.train_sampling_steps, (cfg.batch_size,), device=args.device
                ).long()
            
            grad_norm = None
            with accelerator.accumulate(pixart_decoder):
                optimizer.zero_grad()
            
                T_all = global_context_encoder(Z, mask)
                T_hole, _ = get_hole(T_all, mask, global_context_encoder.num_patches)
            
                loss_term = train_diffusion.training_losses(
                    model=pixart_decoder,
                    x_start=X,
                    timestep=timestep,
                    model_kwargs={
                        "t_hole": T_hole,
                        "mask": mask,
                        "object_features": object_features,
                        "padded_input": True,
                    }
                )

                loss = loss_term['loss'].mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        list(pixart_decoder.parameters()) + list(global_context_encoder.parameters()),
                        cfg.gradient_clip
                    )
                    
                optimizer.step()

            logs = {
                "train/loss": loss.item(),
            }

            if grad_norm is not None:
                logs["train/grad_norm"] = grad_norm.item()

            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})
            accelerator.log(logs, step=global_step)

            if ((epoch - 1) * len(train_dataloader) + step + 1) % cfg.save_model_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    os.umask(0o000)
                    save_checkpoint(os.path.join(cfg.work_dir, 'checkpoints'),
                                    epoch=epoch,
                                    step=(epoch - 1) * len(train_dataloader) + step + 1,
                                    global_context_encoder=accelerator.unwrap_model(global_context_encoder),
                                    pixart_decoder=accelerator.unwrap_model(pixart_decoder),
                                    optimizer=optimizer,
                                    )
                
            break
        break
            
    print("DONE!")

if __name__ == "__main__":
    args = parse_args()
    cfg = OmegaConf.load(args.config_path)

    main(args, cfg.train)