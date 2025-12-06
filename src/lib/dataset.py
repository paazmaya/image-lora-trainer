from pathlib import Path
from typing import Optional
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class LocalImageDataset(Dataset):
    """Base dataset for loading local images with CLIP tokenization (SD/SDXL)."""

    def __init__(
        self,
        image_paths: list[Path],
        tokenizer,
        instance_prompt: str,
        resolution: int = 512,
        preprocessor: Optional[object] = None,
        tokenizer_2: Optional[object] = None,
    ):
        self.image_paths = image_paths
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.instance_prompt = instance_prompt
        self.resolution = resolution
        self.preprocessor = preprocessor

        # Transforms to convert PIL image to tensor and normalize
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")

        # Apply preprocessing (crop and resize) if preprocessor is provided
        if self.preprocessor:
            image = self.preprocessor.process_image(image)
            # If process_image returns None, skip this image
            # (though this shouldn't happen since we validated earlier)
            if image is None:
                # Fallback to next image or raise error
                # For now, we'll just use a blank image
                image = Image.new("RGB", (self.resolution, self.resolution), (0, 0, 0))
        else:
            # No preprocessor, just resize
            image = image.resize(
                (self.resolution, self.resolution), Image.Resampling.LANCZOS
            )

        # Convert to tensor and normalize
        pixel_values = self.image_transforms(image)

        # Tokenize prompt
        input_ids = self.tokenizer(
            self.instance_prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        result = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
        }

        # Add second tokenizer output for SDXL
        if self.tokenizer_2 is not None:
            input_ids_2 = self.tokenizer_2(
                self.instance_prompt,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer_2.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
            result["input_ids_2"] = input_ids_2

        return result
