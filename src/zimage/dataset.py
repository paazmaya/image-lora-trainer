from pathlib import Path
from typing import Optional, TYPE_CHECKING

from torch.utils.data import Dataset
from torchvision import transforms

if TYPE_CHECKING:
    from lib.preprocess import ImagePreprocessor


class ZImageDataset(Dataset):
    """Dataset for Z-Image training with T5 tokenizer."""

    def __init__(
        self,
        image_paths: list[Path],
        tokenizer,
        instance_prompt: str,
        resolution: int = 1024,
        preprocessor: Optional["ImagePreprocessor"] = None,
        max_length: int = 256,
    ):
        self.image_paths = image_paths
        self.tokenizer = tokenizer
        self.instance_prompt = instance_prompt
        self.resolution = resolution
        self.preprocessor = preprocessor
        self.max_length = max_length

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image

        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")

        if self.preprocessor is not None:
            image = self.preprocessor.process_image(image)
            if image is None:
                image = Image.new("RGB", (self.resolution, self.resolution), (0, 0, 0))
        else:
            image = image.resize(
                (self.resolution, self.resolution), Image.Resampling.LANCZOS
            )

        pixel_values = self.image_transforms(image)

        # T5 tokenizer for Z-Image
        text_inputs = self.tokenizer(
            self.instance_prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs.input_ids[0],
            "attention_mask": text_inputs.attention_mask[0],
        }
