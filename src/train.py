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

from dataset import LocalImageDataset
from preprocess import ImagePreprocessor

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
    # Check for GPU availability - training on CPU is impractically slow for diffusion models
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for training. CPU training is too slow for diffusion models.\n"
            "Please ensure you have:\n"
            "1. A CUDA-compatible GPU installed\n"
            "2. CUDA drivers installed\n"
            "3. PyTorch with CUDA support installed\n"
            "Check GPU availability with: python -c 'import torch; print(torch.cuda.is_available())'"
        )
    
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    
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

    # Determine if base_model is a local .safetensors file or a HF repo
    base_model_path = Path(base_model)
    is_local_safetensors = base_model_path.exists() and base_model_path.suffix == ".safetensors"
    
    if is_local_safetensors:
        logger.info(f"Detected local .safetensors file: {base_model}")
        # For local .safetensors files, we'll load the weights from the file
        # and only download small config files from HF
        # Assuming SDXL for .safetensors files (most common case)
        component_repo = "stabilityai/stable-diffusion-xl-base-1.0"
        logger.info(f"Loading configs from {component_repo}, weights from local file")
        
        # Load the full state dict once
        from safetensors.torch import load_file
        logger.info(f"Loading weights from {base_model}")
        full_state_dict = load_file(str(base_model_path))
        logger.info(f"Loaded {len(full_state_dict)} keys from .safetensors file")
        
        model_source = component_repo
    else:
        # It's a HF repo ID or a local directory with the full model structure
        model_source = base_model
        full_state_dict = None

    # Load Tokenizer (small, config only)
    tokenizer = AutoTokenizer.from_pretrained(
        model_source, subfolder="tokenizer", use_fast=False
    )
    
    # Load Scheduler (small, config only)
    noise_scheduler = DDPMScheduler.from_pretrained(model_source, subfolder="scheduler")

    # Load Text Encoder
    if is_local_safetensors:
        logger.info("Loading text encoder config and weights from local file")
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTextConfig
        
        # Download only the config (small JSON file)
        config = CLIPTextConfig.from_pretrained(
            component_repo, subfolder="text_encoder"
        )
        # Initialize model from config without weights
        text_encoder = CLIPTextModel(config)
        
        # Extract text encoder weights from state dict
        text_encoder_keys = [k for k in full_state_dict.keys() if "text_encoder" in k or "cond_stage_model" in k]
        if text_encoder_keys:
            # Map keys from checkpoint format to HF format
            te_state_dict = {}
            for k in text_encoder_keys:
                new_key = k.replace("cond_stage_model.", "").replace("text_encoder.", "")
                te_state_dict[new_key] = full_state_dict[k]
            
            text_encoder.load_state_dict(te_state_dict, strict=False)
            logger.info(f"Loaded text encoder weights ({len(text_encoder_keys)} keys)")
        else:
            logger.warning("No text encoder weights found in local file, using random initialization")
    else:
        text_encoder = CLIPTextModel.from_pretrained(model_source, subfolder="text_encoder")
    
    text_encoder.requires_grad_(False)

    # Load second text encoder for SDXL
    text_encoder_2 = None
    tokenizer_2 = None
    if is_local_safetensors or "xl" in model_source.lower():
        logger.info("Loading second text encoder for SDXL")
        if is_local_safetensors:
            from transformers import CLIPTextModelWithProjection
            config_2 = CLIPTextConfig.from_pretrained(
                component_repo, subfolder="text_encoder_2"
            )
            text_encoder_2 = CLIPTextModelWithProjection(config_2)
            
            # Try to load weights for text_encoder_2
            te2_keys = [k for k in full_state_dict.keys() if "text_encoder_2" in k or "conditioner" in k]
            if te2_keys:
                te2_state_dict = {}
                for k in te2_keys:
                    new_key = k.replace("text_encoder_2.", "").replace("conditioner.", "")
                    te2_state_dict[new_key] = full_state_dict[k]
                text_encoder_2.load_state_dict(te2_state_dict, strict=False)
                logger.info(f"Loaded text_encoder_2 weights ({len(te2_keys)} keys)")
            else:
                logger.warning("No text_encoder_2 weights found in local file, using random initialization")
        else:
            from transformers import CLIPTextModelWithProjection
            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                model_source, subfolder="text_encoder_2"
            )
        
        text_encoder_2.requires_grad_(False)
        
        # Load second tokenizer
        tokenizer_2 = AutoTokenizer.from_pretrained(
            model_source if not is_local_safetensors else component_repo,
            subfolder="tokenizer_2",
            use_fast=False
        )

    # Load VAE
    if is_local_safetensors:
        logger.info("Loading VAE config and weights from local file")
        
        # Download only the config (small JSON file)
        config = AutoencoderKL.load_config(
            component_repo, subfolder="vae"
        )
        # Initialize model from config without weights
        vae = AutoencoderKL.from_config(config)
        
        # Extract VAE weights from state dict
        vae_keys = [k for k in full_state_dict.keys() if "vae" in k or "first_stage_model" in k]
        if vae_keys:
            # Map keys from checkpoint format to HF format
            vae_state_dict = {}
            for k in vae_keys:
                new_key = k.replace("first_stage_model.", "").replace("vae.", "")
                vae_state_dict[new_key] = full_state_dict[k]
            
            vae.load_state_dict(vae_state_dict, strict=False)
            logger.info(f"Loaded VAE weights ({len(vae_keys)} keys)")
        else:
            logger.warning("No VAE weights found in local file, using random initialization")
    else:
        vae = AutoencoderKL.from_pretrained(model_source, subfolder="vae")
    
    vae.requires_grad_(False)

    # Load UNet
    if is_local_safetensors:
        logger.info("Loading UNet config and weights from local file")
        
        # Download only the config (small JSON file)
        config = UNet2DConditionModel.load_config(
            component_repo, subfolder="unet"
        )
        # Initialize model from config without weights
        unet = UNet2DConditionModel.from_config(config)
        
        # Extract UNet weights from state dict
        # Common prefixes in different checkpoint formats
        unet_keys = [k for k in full_state_dict.keys() 
                    if any(prefix in k for prefix in ["model.diffusion_model", "unet", "model.model"])]
        
        if unet_keys:
            # Map keys from checkpoint format to HF format
            unet_state_dict = {}
            for k in unet_keys:
                new_key = k.replace("model.diffusion_model.", "").replace("unet.", "").replace("model.model.", "")
                unet_state_dict[new_key] = full_state_dict[k]
            
            unet.load_state_dict(unet_state_dict, strict=False)
            logger.info(f"Loaded UNet weights ({len(unet_keys)} keys)")
        else:
            # If no prefixed keys found, the entire file might be just UNet weights
            logger.warning("No UNet-specific keys found, attempting to load entire state dict as UNet")
            unet.load_state_dict(full_state_dict, strict=False)
    else:
        # Load UNet from HF repo or local directory
        load_kwargs = {}
        if use_qlora:
            load_kwargs = {
                "load_in_4bit": True,
                "quantization_config": None,
                "device_map": "auto",
            }

        unet = UNet2DConditionModel.from_pretrained(
            model_source, subfolder="unet", **load_kwargs
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

    # Validate Images (in-memory, no disk I/O)
    preprocessor = ImagePreprocessor(resolution=resolution, crop_focus=crop_focus)
    stats = preprocessor.validate_folder(input_dir)

    # Save log
    import json

    with open(Path(output_dir) / "training_log.json", "w") as f:
        json.dump(stats, f, indent=4)

    valid_image_paths = [Path(p) for p in stats["trained"]]

    if not valid_image_paths:
        raise ValueError(
            f"No valid images found in {input_dir}. Check training_log.json for details."
        )

    # Dataset and Dataloader (preprocessing happens on-the-fly)
    train_dataset = LocalImageDataset(
        image_paths=valid_image_paths,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        instance_prompt=instance_prompt,
        resolution=resolution,
        preprocessor=preprocessor,
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
    if text_encoder_2 is not None:
        text_encoder_2.to(accelerator.device, dtype=weight_dtype)

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
                
                # For SDXL, concatenate outputs from both text encoders
                added_cond_kwargs = None
                if text_encoder_2 is not None and hasattr(unet.config, "addition_embed_type") and unet.config.addition_embed_type == "text_time":
                    # Get hidden states from second text encoder
                    encoder_hidden_states_2 = text_encoder_2(batch["input_ids_2"], output_hidden_states=False)[0]
                    
                    # Concatenate the two text encoder outputs
                    encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_2], dim=-1)
                    
                    # Get pooled embeddings from text_encoder_2 (CLIPTextModelWithProjection)
                    pooled_embeds = text_encoder_2(batch["input_ids_2"], output_hidden_states=False).text_embeds
                    
                    # Create time_ids for SDXL (original_size, crops_coords_top_left, target_size)
                    # Using default values for now
                    time_ids = torch.tensor(
                        [[resolution, resolution, 0, 0, resolution, resolution]],
                        device=encoder_hidden_states.device,
                        dtype=torch.long
                    ).repeat(encoder_hidden_states.shape[0], 1)
                    
                    added_cond_kwargs = {
                        "text_embeds": pooled_embeds,
                        "time_ids": time_ids
                    }

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
                    noisy_latents, 
                    timesteps, 
                    encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs
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
        # Save LoRA weights as safetensors
        unet.save_pretrained(output_dir, safe_serialization=True)
        logger.info(f"Model saved to {output_dir}")

