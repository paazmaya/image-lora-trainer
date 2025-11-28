import logging
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, CLIPTextModel
from tqdm.auto import tqdm

from src.dataset import LocalImageDataset
from src.preprocess import ImagePreprocessor

logger = logging.getLogger(__name__)


def train_lora(
    input_dir: str,
    output_dir: str,
    base_model: str,
    resolution: int,
    instance_prompt: str,
    crop_focus: Optional[str],
    use_qlora: bool,
    num_train_epochs: int,
    max_train_steps: int,
    learning_rate: float = 1e-4,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
):
    accelerator_project_config = ProjectConfiguration(
        project_dir=output_dir, logging_dir=os.path.join(output_dir, "logs")
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="fp16",
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Load Tokenizer & Text Encoder
    # Assumes SD 1.5 structure for now. For SDXL, this needs adaptation.
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, subfolder="tokenizer", use_fast=False
    )
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
    text_encoder.requires_grad_(False)

    # Load Scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    # Load VAE
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    vae.requires_grad_(False)

    # Load UNet
    # For QLoRA, we need to load in 4bit if requested
    load_kwargs = {}
    if use_qlora:
        load_kwargs = {
            "load_in_4bit": True,
            "quantization_config": None,
            "device_map": "auto",
        }

    unet = UNet2DConditionModel.from_pretrained(
        base_model, subfolder="unet", **load_kwargs
    )

    if use_qlora:
        unet = prepare_model_for_kbit_training(unet)

    # Freeze UNet parameters
    unet.requires_grad_(False)

    # Add LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)

    if accelerator.is_main_process:
        unet.print_trainable_parameters()

    # Optimizer
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    # Preprocess Images
    preprocessor = ImagePreprocessor(resolution=resolution, crop_focus=crop_focus)
    processed_dir = Path(output_dir) / "processed_images"
    stats = preprocessor.process_folder(input_dir, processed_dir)

    # Save log
    import json

    with open(Path(output_dir) / "training_log.json", "w") as f:
        json.dump(stats, f, indent=4)

    processed_files = [Path(p) for p in stats["trained"]]

    if not processed_files:
        raise ValueError(
            f"No valid images found in {input_dir}. Check training_log.json for details."
        )

    # Dataset and Dataloader
    train_dataset = LocalImageDataset(
        image_paths=processed_files,
        tokenizer=tokenizer,
        instance_prompt=instance_prompt,
        resolution=resolution,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Prepare with Accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # Move VAE and Text Encoder to device
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Training Loop
    if num_train_epochs is None:
        num_train_epochs = math.ceil(max_train_steps / len(train_dataloader))

    logger.info(f"Starting training for {num_train_epochs} epochs")

    global_step = 0
    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")

    for epoch in range(num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            if global_step >= max_train_steps:
                break

    # Save the model
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        # Save LoRA weights
        # We need to construct the output name as requested: {base_model_name}_{folder_basename}
        # But here we just save to output_dir, the caller (CLI) should handle the naming of the directory or file.
        # PEFT saves adapter_model.bin and adapter_config.json

        unet.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")
