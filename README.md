# Image LoRA Trainer

A CLI tool to train LoRA adapters for text-to-image models using a folder of images with intelligent content-aware cropping.

![Stick figure sorting images next to a barrel](./logo.png)

## How It Works

```mermaid
flowchart TD
    Start(["📁 Input Images"]) --> Load["Load Images"]
    Load --> Check{"Crop Focus\nSpecified?"}
    
    Check -->|Yes| YOLO["🔍 YOLO11 Detection"]
    Check -->|No| Center["Center Crop"]
    
    YOLO --> Found{"Target\nFound?"}
    Found -->|Yes| Crop["Smart Crop\n+ Padding"]
    Found -->|No| Skip["⏭️ Skip Image"]
    
    Center --> Resize["Resize to\nTarget Resolution"]
    Crop --> Resize
    
    Resize --> Train["🎯 LoRA Training\n(Diffusers + PEFT)"]
    Skip --> Log
    
    Train --> Save["💾 Save LoRA\nWeights"]
    Save --> Log["📊 Generate\ntraining_log.json"]
    
    Log --> End(["✅ Trained LoRA\n+ Logs"])
    
    style Start fill:#E8F4F8,stroke:#2C5F7C,stroke-width:3px,color:#1a1a1a
    style Load fill:#FFF4E6,stroke:#8B6914,stroke-width:2px,color:#1a1a1a
    style Check fill:#F0E6FF,stroke:#6B46C1,stroke-width:2px,color:#1a1a1a
    style YOLO fill:#E6F7FF,stroke:#1E5A8E,stroke-width:2px,color:#1a1a1a
    style Center fill:#FFF0F5,stroke:#8B4789,stroke-width:2px,color:#1a1a1a
    style Found fill:#F0E6FF,stroke:#6B46C1,stroke-width:2px,color:#1a1a1a
    style Crop fill:#E6FFE6,stroke:#2D5F2D,stroke-width:2px,color:#1a1a1a
    style Skip fill:#FFE6E6,stroke:#8B2E2E,stroke-width:2px,color:#1a1a1a
    style Resize fill:#FFF4E6,stroke:#8B6914,stroke-width:2px,color:#1a1a1a
    style Train fill:#E6F7FF,stroke:#1E5A8E,stroke-width:2px,color:#1a1a1a
    style Save fill:#E6FFE6,stroke:#2D5F2D,stroke-width:2px,color:#1a1a1a
    style Log fill:#FFF4E6,stroke:#8B6914,stroke-width:2px,color:#1a1a1a
    style End fill:#E8F4F8,stroke:#2C5F7C,stroke-width:3px,color:#1a1a1a
```

## Features

- **Content-Aware Cropping**: Uses YOLO11 segmentation to automatically detect and crop to specific objects from [the COCO dataset (faces, people, animals, etc.)](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)
- **Smart Filtering**: Automatically skips images that don't contain the target feature
- **Training Logs**: Generates detailed JSON logs of processed, skipped, and failed images
- **LoRA/QLoRA Training**: Full training pipeline using `diffusers` and `peft` with optional quantization
- **Multiple Model Support**: Works with Stable Diffusion 1.5, SDXL, and **Z-Image-Turbo** (fast 8-step DiT model)
- **Z-Image-Turbo Support**: Train LoRAs for the new 6B parameter single-stream DiT model with 8-bit quantization
- **Visual Verification**: Includes a generation script to test your trained LoRA

## Installation

Requires Python 3.13+ and `uv` for dependency management.

```bash
# Clone the repository
git clone <your-repo-url>
cd image-lora-trainer

# Install dependencies
uv sync
```

### GPU Setup (Required)

**This tool requires a CUDA-capable GPU.** Training on CPU is impractically slow for diffusion models.

1. **Verify you have a CUDA-capable NVIDIA GPU**

2. **Install CUDA 13 drivers** from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)

3. **Install PyTorch with CUDA support:**

   ```bash
   uv pip uninstall torch torchvision -y
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
   ```

   Its about 1.73 GB to download.

4. **Verify GPU is detected:**

   ```bash
   uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
   ```

   You should see something like:
   
   ```
   CUDA available: True
   GPU: NVIDIA GeForce RTX 4070 Ti
   ```

   If you see `CUDA available: False`, the CPU-only version of PyTorch is installed. Follow step 3 above.

## Quick Start

### 1. Prepare Your Images

Place your training images in a folder:
```
my_images/
├── photo1.jpg
├── photo2.png
└── photo3.jpg
```

### 2. Train a LoRA

Basic training:
```bash
uv run python src/main.py --input-dir my_images --base-model runwayml/stable-diffusion-v1-5
```

With content-aware cropping (only trains on images with faces):
```bash
uv run python src/main.py \
  --input-dir my_images \
  --base-model runwayml/stable-diffusion-v1-5 \
  --crop-focus person \
  --resolution 512 \
  --steps 1000
```

With QLoRA (4-bit quantization for lower memory usage):
```bash
uv run python src/main.py \
  --input-dir my_images \
  --base-model stabilityai/stable-diffusion-xl-base-1.0 \
  --use-qlora \
  --resolution 1024
```

### Training Z-Image-Turbo LoRA

[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) is a fast 6B parameter diffusion transformer that produces high-quality images in just 8 steps.

Basic Z-Image training:
```bash
uv run python src/main.py train-zimage \
  --input-dir my_images \
  --instance-prompt "a photo of sks person"
```

With 8-bit quantization (for lower VRAM usage, ~12GB instead of ~24GB):
```bash
uv run python src/main.py train-zimage \
  --input-dir my_images \
  --instance-prompt "a photo of sks person" \
  --use-8bit \
  --steps 500
```

With all options:
```bash
uv run python src/main.py train-zimage \
  --input-dir my_images \
  --instance-prompt "a photo of sks karate practitioner" \
  --crop-focus person \
  --use-8bit \
  --lr 1e-5 \
  --lora-rank 16 \
  --steps 1000
```

With locally available model:


Learning rate: 1e-5 (lower than SD/SDXL)
LoRA rank: 16 (can go higher for more capacity)
Steps: 500-1500 for characters/styles

```ps1
uv run python src/main.py train-zimage --base-model "H:\z-image-turbo" --input-dir "C:\Users\Jukka\Dropbox\Karatejukka 2023\" --instance-prompt "karatejukka" --use-8bit --output-dir "C:\Users\Jukka\Dropbox\Karatejukka-2023-Z-image-turbo-lora\"
```

**Note:** Z-Image-Turbo uses a [training adapter](https://huggingface.co/ostris/zimage_turbo_training_adapter) by default to prevent the distillation from breaking during training. This is recommended for short training runs (styles, concepts, characters).

### 3. Check Training Results

After training, check the `training_log.json` in your output directory:
```json
{
  "base_folder": "/absolute/path/to/my_images",
  "trained": ["image1.png", "image2.png"],
  "skipped": ["image3.png"],
  "failed": []
}
```

### 4. Generate Images with Your LoRA

For Stable Diffusion:
```bash
uv run python src/generate.py sd \
  --base-model runwayml/stable-diffusion-v1-5 \
  --lora-path stable-diffusion-v1-5_my_images \
  --prompt "a photo of a sks person" \
  --output result.png
```

For Z-Image-Turbo:
```bash
uv run python src/generate.py zimage \
  --lora-path zimage-turbo_my_images \
  --prompt "a photo of sks person, professional studio lighting" \
  --output result.png
```

**Important:** Use the same trigger word ("sks" in this example) that you specified in `--instance-prompt` during training.

More generation examples:
```bash
# Portrait with different styling
uv run python src/generate.py --lora-path <path> --prompt "portrait of sks person, oil painting"

# Different context
uv run python src/generate.py --lora-path <path> --prompt "sks person in a futuristic city"
```

## CLI Options

### Training SD/SDXL (`src/main.py train`)

| Option | Description | Default |
|--------|-------------|---------|
| `--input-dir` | Path to training images | Required |
| `--output-dir` | Output directory for LoRA | Current directory |
| `--base-model` | Hugging Face model ID or local path | `runwayml/stable-diffusion-v1-5` |
| `--resolution` | Training image resolution | 512 |
| `--crop-focus` | Object to focus on (e.g., "person", "face", "dog") | None (center crop) |
| `--use-qlora` | Enable 4-bit quantization | False |
| `--instance-prompt` | Training prompt with trigger word | "a photo of a sks person" |
| `--steps` | Number of training steps | 1000 |
| `--epochs` | Number of epochs (overrides steps) | None |

### Training Z-Image-Turbo (`src/main.py train-zimage`)

| Option | Description | Default |
|--------|-------------|---------|
| `--input-dir` | Path to training images | Required |
| `--output-dir` | Output directory for LoRA | Current directory |
| `--base-model` | Z-Image model ID | `Tongyi-MAI/Z-Image-Turbo` |
| `--resolution` | Training image resolution | 1024 |
| `--crop-focus` | Object to focus on | None (center crop) |
| `--use-8bit` | Enable 8-bit quantization | False |
| `--no-training-adapter` | Disable de-distillation adapter | False (adapter enabled) |
| `--instance-prompt` | Training prompt with trigger word | "a photo of a sks person" |
| `--steps` | Number of training steps | 1000 |
| `--lr` | Learning rate | 1e-5 |
| `--lora-rank` | LoRA rank | 16 |
| `--lora-alpha` | LoRA alpha | 16 |
| `--save-steps` | Save checkpoint every N steps | 500 |

**About `--instance-prompt`:**
The instance prompt contains a **trigger word** (like "sks") that the model learns to associate with your training images. This trigger word is what you'll use later when generating images with the LoRA.

- Use a unique, uncommon token (e.g., "sks", "xyz", "abc123")
- Include the class name (e.g., "person", "dog", "style")
- Example: `"a photo of sks person"` → Use `"sks person"` in generation prompts

### Generation (`src/generate.py sd` / `src/generate.py zimage`)

**SD Generation Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--base-model` | Base model ID or path | `runwayml/stable-diffusion-v1-5` |
| `--lora-path` | Path to trained LoRA | Required |
| `--prompt` | Generation prompt | Required |
| `--output` | Output filename | `output.png` |
| `--steps` | Inference steps | 30 |

**Z-Image Generation Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--base-model` | Z-Image model ID | `Tongyi-MAI/Z-Image-Turbo` |
| `--lora-path` | Path to trained LoRA | Required |
| `--prompt` | Generation prompt | Required |
| `--output` | Output filename | `output.png` |
| `--width` | Image width | 1024 |
| `--height` | Image height | 1024 |
| `--steps` | Inference steps | 8 |
| `--seed` | Random seed | None (random) |
| `--lora-scale` | LoRA weight scale | 1.0 |

## Content-Aware Cropping

When you specify `--crop-focus`, the tool uses [YOLO11](https://docs.ultralytics.com/models/yolo11/) to detect objects in your images:

- **Supported objects**: Any object in the COCO dataset (person, dog, cat, car, etc.)
- **Behavior**: Images without the target object are automatically skipped
- **Fallback**: If no focus is specified, images are center-cropped

Example crop focuses:
- `person` - Crops to people
- `face` - Crops to faces (use "person" for full body)
- `dog`, `cat` - Crops to animals
- `car`, `truck` - Crops to vehicles

## Development

### Running Tests

```bash
uv run pytest tests/
```

### Linting and Formatting

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Output Structure

After training, your output directory will contain:
```
stable-diffusion-v1-5_my_images/
├── adapter_config.json       # LoRA configuration
├── adapter_model.safetensors # LoRA weights
├── training_log.json         # Processing log
├── logs/                     # Training logs
└── processed_images/         # Preprocessed images
```

## Requirements

- Python 3.13+
- CUDA-capable GPU (required)
- **Stable Diffusion 1.5**: ~8GB VRAM (less with QLoRA)
- **SDXL**: ~16GB VRAM (less with QLoRA)
- **Z-Image-Turbo**: ~24GB VRAM (or ~12GB with 8-bit quantization)

## License

[Your License Here]

## Acknowledgments

- Built with [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- Uses [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- LoRA implementation via [PEFT](https://github.com/huggingface/peft)
- Z-Image-Turbo by [Tongyi-MAI](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- Z-Image training adapter by [ostris](https://huggingface.co/ostris/zimage_turbo_training_adapter)
