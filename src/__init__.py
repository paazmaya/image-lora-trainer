"""
Image LoRA Trainer - Train LoRA adapters for various diffusion models.

Supports:
- Stable Diffusion (SD 1.x, 2.x)
- Stable Diffusion XL (SDXL)
- Z-Image-Turbo

Each model type has its own submodule:
- lib: Common utilities (preprocessing, base dataset)
- stable_diffusion: SD/SDXL training and generation
- zimage: Z-Image-Turbo training and generation
"""

from lib import ImagePreprocessor, LocalImageDataset
from stable_diffusion import train_lora, generate_sd
from zimage import train_zimage_lora, generate_zimage, ZImageDataset

__all__ = [
    # Common
    "ImagePreprocessor",
    "LocalImageDataset",
    # Stable Diffusion
    "train_lora",
    "generate_sd",
    # Z-Image
    "train_zimage_lora",
    "generate_zimage",
    "ZImageDataset",
]
