from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class LocalImageDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        tokenizer,
        instance_prompt: str,
        resolution: int = 512,
        center_crop: bool = False,  # We already pre-processed, but just in case
    ):
        self.image_paths = image_paths
        self.tokenizer = tokenizer
        self.instance_prompt = instance_prompt
        self.resolution = resolution

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    resolution, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(resolution)
                if center_crop
                else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")

        # Apply transforms
        pixel_values = self.image_transforms(image)

        # Tokenize prompt
        # Note: This assumes a single prompt for all images for now (DreamBooth style)
        input_ids = self.tokenizer(
            self.instance_prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
        }
