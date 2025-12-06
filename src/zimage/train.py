"""
Training script for Z-Image-Turbo LoRA with 8-bit quantization support.

Z-Image-Turbo uses a Single-Stream DiT (S3-DiT) architecture with:
- ZImageTransformer2DModel (transformer backbone, not UNet)
- T5 text encoder (not CLIP)
- Flow matching scheduler

Key consideration: Z-Image-Turbo is a distilled model. Training directly on it
breaks the distillation. We use ostris/zimage_turbo_training_adapter during
training to prevent this, then remove it during inference.
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm

from lib.preprocess import ImagePreprocessor
from .dataset import ZImageDataset

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Z-Image specific constants
ZIMAGE_TURBO_MODEL = "Tongyi-MAI/Z-Image-Turbo"
ZIMAGE_TRAINING_ADAPTER = "ostris/zimage_turbo_training_adapter"


def train_zimage_lora(
    input_dir: str,
    output_dir: str,
    base_model: str = ZIMAGE_TURBO_MODEL,
    resolution: int = 1024,
    instance_prompt: str = "a photo of a sks person",
    crop_focus: Optional[str] = None,
    use_8bit: bool = False,
    use_training_adapter: bool = True,
    num_train_epochs: Optional[int] = None,
    max_train_steps: int = 1000,
    learning_rate: float = 1e-5,  # Lower LR recommended for Z-Image
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    save_steps: int = 500,
):
    """
    Train a LoRA adapter for Z-Image-Turbo.

    Args:
        input_dir: Path to training images
        output_dir: Output directory for trained LoRA
        base_model: Z-Image model ID (default: Tongyi-MAI/Z-Image-Turbo)
        resolution: Training resolution (default: 1024)
        instance_prompt: Training prompt
        crop_focus: Object to focus crop on (e.g., 'face', 'person')
        use_8bit: Use 8-bit quantization for lower VRAM usage
        use_training_adapter: Use the de-distillation training adapter (recommended)
        num_train_epochs: Number of epochs (optional)
        max_train_steps: Maximum training steps
        learning_rate: Learning rate (1e-5 recommended for Z-Image)
        train_batch_size: Batch size
        gradient_accumulation_steps: Gradient accumulation steps
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        save_steps: Save checkpoint every N steps
    """
    # Check for GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for training. CPU training is too slow for diffusion models.\n"
            "Please ensure you have:\n"
            "1. A CUDA-compatible GPU installed\n"
            "2. CUDA drivers installed\n"
            "3. PyTorch with CUDA support installed"
        )

    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(
        f"Training Z-Image-Turbo LoRA with {'8-bit' if use_8bit else 'bf16'} precision"
    )

    # Setup accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=output_dir, logging_dir=os.path.join(output_dir, "logs")
    )

    # Use bf16 for Z-Image (it was trained with bf16)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="bf16",
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Import Z-Image components (requires diffusers from source)
    try:
        from diffusers import ZImagePipeline
    except ImportError:
        raise ImportError(
            "Z-Image support requires the latest diffusers from source.\n"
            "Install with: pip install git+https://github.com/huggingface/diffusers"
        )

    logger.info(f"Loading Z-Image pipeline from {base_model}")

    # Load the full pipeline first to get all components
    pipe = ZImagePipeline.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
    )

    # Extract components
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    transformer = pipe.transformer

    # Clean up pipeline to free memory
    del pipe
    torch.cuda.empty_cache()

    # Convert all transformer buffers to bfloat16 to avoid dtype mismatches
    # The ZImageTransformer has some buffers (like x_pad_token, cap_pad_token)
    # that may remain float32 and cause issues during forward pass
    def convert_buffers_to_bf16(module):
        for name, buf in list(module.named_buffers(recurse=False)):
            if buf is not None and buf.dtype == torch.float32:
                # Re-register the buffer with the correct dtype
                module.register_buffer(name, buf.to(torch.bfloat16))
        for child in module.children():
            convert_buffers_to_bf16(child)

    convert_buffers_to_bf16(transformer)

    # Freeze text encoder and VAE
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    # Apply 8-bit quantization if requested
    if use_8bit:
        try:
            import bitsandbytes as bnb
            from diffusers.models import ZImageTransformer2DModel

            logger.info("Applying 8-bit quantization to transformer")
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )

            # Delete the existing transformer and reload with quantization
            del transformer
            torch.cuda.empty_cache()

            # Load transformer directly with 8-bit quantization
            # The transformer subfolder contains the model
            transformer_path = Path(base_model) / "transformer"
            if not transformer_path.exists():
                # If using HuggingFace model ID, load from hub
                transformer = ZImageTransformer2DModel.from_pretrained(
                    base_model,
                    subfolder="transformer",
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )
            else:
                # Local path
                transformer = ZImageTransformer2DModel.from_pretrained(
                    str(transformer_path),
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )

            # Convert buffers to bf16 for the 8-bit model
            convert_buffers_to_bf16(transformer)

            # Note: We skip prepare_model_for_kbit_training because ZImageTransformer
            # doesn't have get_input_embeddings method. The model will work fine without it
            # for LoRA training since we're only training the adapter weights.
            logger.info("8-bit quantization applied successfully")

        except (ImportError, Exception) as e:
            logger.warning(
                f"8-bit quantization failed: {e}. Continuing without quantization."
            )
            use_8bit = False
            # Reload transformer without quantization
            pipe = ZImagePipeline.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
            )
            transformer = pipe.transformer
            del pipe
            torch.cuda.empty_cache()
            # Convert buffers to bf16
            convert_buffers_to_bf16(transformer)

    # Load training adapter if requested (highly recommended for distilled models)
    training_adapter_loaded = False
    if use_training_adapter:
        try:
            logger.info(f"Loading training adapter from {ZIMAGE_TRAINING_ADAPTER}")
            # Load the de-distillation adapter

            # First, we need to add a LoRA config to make it PEFT-compatible
            # The training adapter is itself a LoRA, we load it as a base
            transformer.load_adapter(
                ZIMAGE_TRAINING_ADAPTER, adapter_name="training_adapter"
            )
            transformer.set_adapter("training_adapter")
            training_adapter_loaded = True
            logger.info("Training adapter loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load training adapter: {e}")
            logger.warning(
                "Training without adapter - distillation may degrade over long runs"
            )

    # Freeze transformer base parameters
    transformer.requires_grad_(False)

    # Z-Image transformer target modules for LoRA
    # Based on the S3-DiT architecture, we target attention layers
    zimage_target_modules = [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        # Also target feedforward for better results
        "ff.net.0.proj",
        "ff.net.2",
    ]

    # Add LoRA to transformer
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=zimage_target_modules,
        lora_dropout=0.05,
        bias="none",
    )

    transformer = get_peft_model(transformer, lora_config)

    if accelerator.is_main_process:
        transformer.print_trainable_parameters()

    # Optimizer - use 8-bit Adam if available and 8-bit mode is on
    if use_8bit:
        try:
            import bitsandbytes as bnb

            optimizer_cls = bnb.optim.Adam8bit  # type: ignore[attr-defined]
            logger.info("Using 8-bit Adam optimizer")
        except ImportError:
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    # Validate images
    preprocessor = ImagePreprocessor(resolution=resolution, crop_focus=crop_focus)
    stats = preprocessor.validate_folder(input_dir)

    # Save log
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "training_log.json", "w") as f:
        json.dump(stats, f, indent=4)

    valid_image_paths = [Path(p) for p in stats["trained"]]

    if not valid_image_paths:
        raise ValueError(
            f"No valid images found in {input_dir}. Check training_log.json for details."
        )

    # Dataset and Dataloader
    train_dataset = ZImageDataset(
        image_paths=valid_image_paths,
        tokenizer=tokenizer,
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

    # Scheduler for training
    from diffusers.optimization import get_scheduler

    lr_scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Prepare with Accelerator
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # Move VAE and Text Encoder to device
    weight_dtype = torch.bfloat16
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Training Loop
    if num_train_epochs is None:
        num_train_epochs = math.ceil(max_train_steps / len(train_dataloader))

    logger.info(
        f"Starting training for {num_train_epochs} epochs ({max_train_steps} steps)"
    )
    logger.info(f"  Batch size: {train_batch_size}")
    logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
    logger.info(
        f"  Effective batch size: {train_batch_size * gradient_accumulation_steps}"
    )
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  LoRA rank: {lora_rank}, alpha: {lora_alpha}")

    global_step = 0
    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")

    for epoch in range(num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Convert images to latent space
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Z-Image uses flow matching, sample timesteps uniformly
                # The pipeline uses (1000 - t) / 1000 transformation
                # For training, we sample in [0, 1] directly
                timesteps = torch.rand(
                    (bsz,), device=latents.device, dtype=weight_dtype
                )

                # Create noisy latents using flow matching interpolation
                # x_t = (1 - t) * x_0 + t * noise
                noisy_latents = (
                    1 - timesteps.view(-1, 1, 1, 1)
                ) * latents + timesteps.view(-1, 1, 1, 1) * noise

                # Get text embeddings from T5
                encoder_output = text_encoder(
                    batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    output_hidden_states=True,
                )
                # Z-Image uses the last hidden state as caption features
                encoder_hidden_states = encoder_output.last_hidden_state

                # Z-Image transformer uses a unique API:
                # - x: List[torch.Tensor] - list of latent tensors (one per sample)
                # - t: timestep tensor
                # - cap_feats: List[torch.Tensor] - list of caption embeddings (one per sample)
                #
                # The target for flow matching is the velocity: v = noise - x_0
                target = noise - latents

                # Prepare inputs in Z-Image's expected format
                # Add frame dimension and convert to list
                noisy_latents_5d = noisy_latents.unsqueeze(2)  # [B, C, 1, H, W]
                noisy_latents_list = list(noisy_latents_5d.unbind(dim=0))

                # Convert caption embeddings to list
                cap_feats_list = list(encoder_hidden_states.unbind(dim=0))

                # Forward pass - Z-Image transformer returns a list of outputs
                model_out_list = transformer(
                    x=noisy_latents_list,
                    t=timesteps,
                    cap_feats=cap_feats_list,
                    return_dict=False,
                )[0]

                # Convert output list back to tensor for loss calculation
                # Each item in model_out_list is [C, 1, H, W], stack to [B, C, 1, H, W]
                model_pred = torch.stack(model_out_list, dim=0).squeeze(
                    2
                )  # [B, C, H, W]

                # MSE loss on velocity prediction
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item())
                global_step += 1

                # Save checkpoint
                if global_step % save_steps == 0:
                    if accelerator.is_main_process:
                        save_path = Path(output_dir) / f"checkpoint-{global_step}"
                        save_path.mkdir(parents=True, exist_ok=True)
                        unwrapped = accelerator.unwrap_model(transformer)
                        # Only save the new LoRA weights, not the training adapter
                        unwrapped.save_pretrained(
                            save_path,
                            safe_serialization=True,
                            selected_adapters=["default"],  # Save only our trained LoRA
                        )
                        logger.info(f"Checkpoint saved to {save_path}")

            if global_step >= max_train_steps:
                break

        if global_step >= max_train_steps:
            break

    # Save final model
    if accelerator.is_main_process:
        transformer = accelerator.unwrap_model(transformer)
        # Save only the trained LoRA adapter (not the training adapter)
        transformer.save_pretrained(
            output_dir,
            safe_serialization=True,
            selected_adapters=["default"],
        )

        # Save training config
        config = {
            "base_model": base_model,
            "resolution": resolution,
            "instance_prompt": instance_prompt,
            "learning_rate": learning_rate,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "max_train_steps": max_train_steps,
            "use_8bit": use_8bit,
            "use_training_adapter": training_adapter_loaded,
        }
        with open(Path(output_dir) / "training_config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Model saved to {output_dir}")
        logger.info(
            "Note: When using this LoRA for inference, do NOT load the training adapter."
        )
        logger.info("Just load your LoRA directly on the base Z-Image-Turbo model.")
