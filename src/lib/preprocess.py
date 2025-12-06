from pathlib import Path
from typing import Optional, Union
from PIL import Image
from ultralytics import YOLO


class ImagePreprocessor:
    def __init__(self, resolution: int = 512, crop_focus: Optional[str] = None):
        self.resolution = resolution
        self.crop_focus = crop_focus
        self.model = None
        if self.crop_focus:
            # Load YOLO11 segmentation model
            # Using yolo11n-seg.pt for speed/efficiency
            self.model = YOLO("yolo11n-seg.pt")

    def validate_folder(self, input_dir: Union[str, Path]) -> dict:
        """
        Validate all images in the input directory without saving them.
        Returns a dictionary with lists of 'trained', 'skipped', and 'failed' file paths.
        Images are validated in-memory to check if they pass the crop_focus filter.
        """
        input_path = Path(input_dir).resolve()

        stats = {
            "base_folder": str(input_path),
            "trained": [],
            "skipped": [],
            "failed": [],
        }

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for file_path in input_path.iterdir():
            if file_path.suffix.lower() in valid_extensions:
                try:
                    img = Image.open(file_path).convert("RGB")

                    # Only validate if crop_focus is set
                    if self.crop_focus and self.model:
                        # Check if the image contains the target object
                        if not self._has_target_object(img):
                            stats["skipped"].append(str(file_path))
                            continue

                    # Image is valid, record the original path
                    stats["trained"].append(str(file_path))

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    stats["failed"].append(str(file_path))

        return stats

    def _has_target_object(self, image: Image.Image) -> bool:
        """
        Check if the image contains the target object for crop_focus.
        Returns True if target found, False otherwise.
        """
        results = self.model(image, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]

                if cls_name.lower() == self.crop_focus.lower():
                    return True

        return False

    def process_image(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Crops and resizes a single image.
        Returns None if the image should be skipped (e.g. target not found).
        """
        if self.crop_focus and self.model:
            image = self._content_aware_crop(image)
            if image is None:
                return None
        else:
            image = self._center_crop(image)

        image = image.resize(
            (self.resolution, self.resolution), Image.Resampling.LANCZOS
        )
        return image

    def _content_aware_crop(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Uses YOLO11 to detect the focus object and crop around it.
        Returns None if target not found.
        """
        # Run inference
        results = self.model(image, verbose=False)

        target_box = None

        # Check if any detections match our focus class
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]

                if cls_name.lower() == self.crop_focus.lower():
                    # Found our target
                    xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    target_box = xyxy
                    break
            if target_box is not None:
                break

        if target_box is None:
            # Target not found, skip image
            return None

        # Calculate crop with aspect ratio preservation (square)
        x1, y1, x2, y2 = target_box
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2

        # We want a square crop that covers the object
        # Size should be at least max(w, h)
        # Add some padding?
        size = max(w, h) * 1.2  # 20% padding

        # Ensure we don't go outside image bounds too much (we can pad later)
        # But for now let's just calculate crop coordinates

        half_size = size / 2
        crop_x1 = max(0, cx - half_size)
        crop_y1 = max(0, cy - half_size)
        crop_x2 = min(image.width, cx + half_size)
        crop_y2 = min(image.height, cy + half_size)

        # If the crop is not square (because of image bounds), we need to pad
        # Or we can just crop what we have and then resize/pad.
        # Let's crop then square-pad.

        cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        return self._square_pad(cropped)

    def _center_crop(self, image: Image.Image) -> Image.Image:
        """
        Standard center crop to square.
        """
        width, height = image.size
        new_size = min(width, height)

        left = (width - new_size) / 2
        top = (height - new_size) / 2
        right = (width + new_size) / 2
        bottom = (height + new_size) / 2

        return image.crop((left, top, right, bottom))

    def _square_pad(self, image: Image.Image) -> Image.Image:
        """
        Pads an image to make it square.
        """
        width, height = image.size
        if width == height:
            return image

        max_dim = max(width, height)
        new_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))

        # Paste center
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        new_img.paste(image, (left, top))
        return new_img

    def process_folder(
        self, input_dir: Union[str, Path], output_dir: Union[str, Path]
    ) -> dict:
        """
        Process all images in input_dir and save to output_dir.
        Returns stats about processed, skipped, and failed files.
        """
        input_path = Path(input_dir).resolve()
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        stats = {
            "base_folder": str(input_path),
            "trained": [],
            "skipped": [],
            "failed": [],
        }

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for file_path in input_path.iterdir():
            if file_path.suffix.lower() in valid_extensions:
                try:
                    img = Image.open(file_path).convert("RGB")
                    processed = self.process_image(img)

                    if processed is None:
                        stats["skipped"].append(str(file_path))
                        continue

                    # Save as PNG
                    output_file = output_path / f"{file_path.stem}.png"
                    processed.save(output_file)
                    stats["trained"].append(str(file_path))

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    stats["failed"].append(str(file_path))

        return stats
