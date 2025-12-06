#!/usr/bin/env python
"""
Convert a diffusers model to reduced precision formats.

This script loads a model from a local directory (downloaded from HuggingFace)
and saves it in a reduced precision format to reduce storage size and memory usage.

Supported formats:
- bf16: bfloat16 (16-bit brain floating point)
- e4m3fn: float8_e4m3fn (8-bit floating point, 4-bit exponent, 3-bit mantissa, higher precision)
- e5m2: float8_e5m2 (8-bit floating point, 5-bit exponent, 2-bit mantissa, wider range)
  See https://onnx.ai/onnx/technical/float8.html for more info on float8 formats.

Usage:
    python scripts/convert_floats.py --input H:/some-model
    python scripts/convert_floats.py --input H:/some-model --dtype e4m3fn
    python scripts/convert_floats.py --input H:/models/model.safetensors
"""

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Literal

import torch
from torch import nn

# Type alias for supported dtypes
DtypeOption = Literal["bf16", "e4m3fn", "e5m2"]

# Mapping from option name to torch dtype
DTYPE_MAP: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "e4m3fn": torch.float8_e4m3fn,
    "e5m2": torch.float8_e5m2,
}

# Mapping from option name to output suffix
SUFFIX_MAP: dict[str, str] = {
    "bf16": "-bf16",
    "e4m3fn": "-e4m3fn",
    "e5m2": "-e5m2",
}

# Bytes per element for each dtype
DTYPE_SIZES: dict[str, int] = {
    "torch.float64": 8,
    "torch.float32": 4,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.float8_e4m3fn": 1,
    "torch.float8_e4m3fnuz": 1,
    "torch.float8_e5m2": 1,
    "torch.float8_e5m2fnuz": 1,
    "torch.int64": 8,
    "torch.int32": 4,
    "torch.int16": 2,
    "torch.int8": 1,
    "torch.uint8": 1,
    "torch.bool": 1,
}


def _get_dtype_size(dtype_str: str) -> int:
    """Get bytes per element for a dtype string."""
    return DTYPE_SIZES.get(dtype_str, 4)  # Default to 4 bytes if unknown


def format_size(num_bytes: int | float) -> str:
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_dtype_distribution(model: nn.Module) -> dict[str, int]:
    """Get distribution of dtypes across model parameters."""
    dtype_counts: Counter[str] = Counter()
    for param in model.parameters():
        dtype_counts[str(param.dtype)] += param.numel()
    return dict(dtype_counts)


def get_layer_info(model: nn.Module) -> dict[str, int]:
    """Count different layer types in the model."""
    layer_counts: Counter[str] = Counter()
    for module in model.modules():
        layer_counts[type(module).__name__] += 1
    return dict(layer_counts)


def get_model_config_info(model: nn.Module) -> dict:
    """Extract configuration info from model if available."""
    info = {}
    if hasattr(model, "config"):
        config = model.config
        # Common config attributes
        for attr in [
            "in_channels",
            "out_channels",
            "sample_size",
            "num_attention_heads",
            "attention_head_dim",
            "num_layers",
            "cross_attention_dim",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "patch_size",
        ]:
            if hasattr(config, attr):
                info[attr] = getattr(config, attr)
    return info


def print_model_info(
    model: nn.Module, model_name: str = "Model", is_input: bool = False
):
    """Print detailed information about a model."""
    label = "INPUT" if is_input else "OUTPUT"
    print(f"\n{'=' * 60}")
    print(f"  [{label}] {model_name} Information")
    print(f"{'=' * 60}")

    # Basic info
    total_params, trainable_params = count_parameters(model)
    print("\nParameters:")
    print(
        f"  Total:     {total_params:,} ({format_size(total_params * 4)})"
    )  # Assuming fp32
    print(f"  Trainable: {trainable_params:,}")

    # Dtype distribution
    dtype_dist = get_dtype_distribution(model)
    print("\nDtype Distribution:")
    total_elements = sum(dtype_dist.values())
    for dtype, count in sorted(dtype_dist.items(), key=lambda x: -x[1]):
        pct = count / total_elements * 100
        print(f"  {dtype}: {count:,} ({pct:.1f}%)")

    # Layer counts
    layer_info = get_layer_info(model)
    # Filter to interesting layer types
    interesting_layers = [
        "Linear",
        "Conv2d",
        "Conv1d",
        "LayerNorm",
        "GroupNorm",
        "Attention",
        "MultiHeadAttention",
        "SelfAttention",
        "CrossAttention",
        "TransformerBlock",
        "BasicTransformerBlock",
        "Transformer2DModelOutput",
    ]
    print("\nLayer Counts (key layers):")
    for layer_type in interesting_layers:
        if layer_type in layer_info:
            print(f"  {layer_type}: {layer_info[layer_type]}")

    # Show total unique layer types
    print(f"  (Total unique layer types: {len(layer_info)})")

    # Config info
    config_info = get_model_config_info(model)
    if config_info:
        print("\nModel Configuration:")
        for key, value in config_info.items():
            print(f"  {key}: {value}")

    # Input/Output shapes if available
    if hasattr(model, "config"):
        config = model.config
        print("\nInput/Output Info:")
        if hasattr(config, "in_channels"):
            print(f"  Input channels: {config.in_channels}")
        if hasattr(config, "out_channels"):
            print(f"  Output channels: {config.out_channels}")
        if hasattr(config, "sample_size"):
            size = config.sample_size
            if isinstance(size, (list, tuple)):
                print(f"  Sample size: {size[0]}x{size[1]}")
            else:
                print(f"  Sample size: {size}x{size}")

    print(f"{'=' * 60}\n")


def print_pipeline_info(pipe, pipeline_type: str, is_input: bool = False):
    """Print information about all components in a pipeline."""
    label = "INPUT" if is_input else "OUTPUT"
    print(f"\n{'#' * 70}")
    print(f"  [{label}] Pipeline: {pipeline_type}")
    print(f"{'#' * 70}")

    # List all components
    print("\nComponents:")
    components = []
    for attr in dir(pipe):
        if attr.startswith("_"):
            continue
        obj = getattr(pipe, attr, None)
        if obj is not None and isinstance(obj, nn.Module):
            components.append((attr, obj))

    for name, component in components:
        print(f"  - {name}: {type(component).__name__}")

    # Print detailed info for main components
    main_components = ["transformer", "unet", "vae", "text_encoder", "text_encoder_2"]
    for name in main_components:
        component = getattr(pipe, name, None)
        if component is not None and isinstance(component, nn.Module):
            print_model_info(
                component, f"{name} ({type(component).__name__})", is_input=is_input
            )


def convert_model(
    input_dir: str, output_dir: str | None = None, dtype: DtypeOption = "bf16"
):
    """
    Convert a diffusers model to reduced precision.

    Args:
        input_dir: Path to the input model directory
        output_dir: Path to save the converted model (defaults to input_dir + suffix)
        dtype: Target dtype - "bf16" for bfloat16, "e4m3fn" for float8_e4m3fn
    """
    input_path = Path(input_dir)
    target_dtype = DTYPE_MAP[dtype]
    suffix = SUFFIX_MAP[dtype]

    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_path}")
        sys.exit(1)

    # Determine output path
    if output_dir:
        output_path = Path(output_dir)
    elif input_path.is_file():
        # For file input: same directory, filename with suffix
        stem = input_path.stem
        output_path = input_path.parent / f"{stem}{suffix}.safetensors"
    else:
        # Default: append dtype suffix to input directory name
        output_path = input_path.parent / f"{input_path.name}{suffix}"

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Target dtype: {target_dtype}")

    # Try to detect the model type and load appropriately
    # Check if input is a file (safetensors or bin)
    if input_path.is_file():
        if input_path.suffix in [".safetensors", ".bin"]:
            convert_single_file(input_path, output_path, target_dtype)
        else:
            print(f"Error: Unsupported file type: {input_path.suffix}")
            print("Supported file types: .safetensors, .bin")
            sys.exit(1)
    elif input_path.is_dir():
        model_index = input_path / "model_index.json"

        if model_index.exists():
            # It's a diffusers pipeline
            convert_pipeline(input_path, output_path, target_dtype)
        elif (input_path / "config.json").exists():
            # It's a single model (transformer, unet, etc.)
            convert_single_model(input_path, output_path, target_dtype)
        elif list(input_path.glob("*.safetensors")) or list(input_path.glob("*.bin")):
            # Directory with safetensors/bin files but no config
            convert_safetensors_files(input_path, output_path, target_dtype)
        else:
            print(
                "Error: Could not detect model type. Expected model_index.json, config.json, or .safetensors/.bin files"
            )
            sys.exit(1)
    else:
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    print("Done!")


def convert_pipeline(
    input_path: Path, output_path: Path, target_dtype: torch.dtype = torch.bfloat16
):
    """Convert a full diffusers pipeline to target dtype."""
    import json

    # Read model_index.json to understand the pipeline structure
    with open(input_path / "model_index.json") as f:
        model_index = json.load(f)

    pipeline_class = model_index.get("_class_name", "Unknown")
    print(f"Detected pipeline: {pipeline_class}")

    # Determine model type for display
    if "ZImage" in pipeline_class:
        model_type = "Z-Image-Turbo (S3-DiT)"
    elif "StableDiffusionXL" in pipeline_class:
        model_type = "Stable Diffusion XL"
    elif "Flux" in pipeline_class:
        model_type = "Flux"
    elif "StableDiffusion" in pipeline_class:
        model_type = "Stable Diffusion 1.x/2.x"
    else:
        model_type = pipeline_class

    # First, load WITHOUT dtype conversion to see original dtypes
    print("\n>>> Loading input model to analyze original dtypes...")
    input_pipe = None

    if "ZImage" in pipeline_class:
        from diffusers import ZImagePipeline

        input_pipe = ZImagePipeline.from_pretrained(
            str(input_path),
            low_cpu_mem_usage=True,
        )
    elif "StableDiffusionXL" in pipeline_class:
        from diffusers import StableDiffusionXLPipeline

        input_pipe = StableDiffusionXLPipeline.from_pretrained(
            str(input_path),
            low_cpu_mem_usage=True,
        )
    elif "StableDiffusion" in pipeline_class:
        from diffusers import StableDiffusionPipeline

        input_pipe = StableDiffusionPipeline.from_pretrained(
            str(input_path),
            low_cpu_mem_usage=True,
        )
    elif "Flux" in pipeline_class:
        from diffusers import FluxPipeline

        input_pipe = FluxPipeline.from_pretrained(
            str(input_path),
            low_cpu_mem_usage=True,
        )
    else:
        from diffusers import DiffusionPipeline

        input_pipe = DiffusionPipeline.from_pretrained(
            str(input_path),
            low_cpu_mem_usage=True,
        )

    # Print input model information
    if input_pipe is not None:
        print_pipeline_info(input_pipe, model_type, is_input=True)
        del input_pipe
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Now load and convert to target dtype
    print(f"\n>>> Converting to {target_dtype}...")

    # For bf16, we can use torch_dtype parameter directly
    # For float8 types, diffusers doesn't support direct loading, so we use safetensors conversion
    if target_dtype == torch.bfloat16:
        pipe = None

        if "ZImage" in pipeline_class:
            from diffusers import ZImagePipeline

            pipe = ZImagePipeline.from_pretrained(
                str(input_path),
                torch_dtype=target_dtype,
                low_cpu_mem_usage=True,
            )
        elif "StableDiffusionXL" in pipeline_class:
            from diffusers import StableDiffusionXLPipeline

            pipe = StableDiffusionXLPipeline.from_pretrained(
                str(input_path),
                torch_dtype=target_dtype,
                low_cpu_mem_usage=True,
            )
        elif "StableDiffusion" in pipeline_class:
            from diffusers import StableDiffusionPipeline

            pipe = StableDiffusionPipeline.from_pretrained(
                str(input_path),
                torch_dtype=target_dtype,
                low_cpu_mem_usage=True,
            )
        elif "Flux" in pipeline_class:
            from diffusers import FluxPipeline

            pipe = FluxPipeline.from_pretrained(
                str(input_path),
                torch_dtype=target_dtype,
                low_cpu_mem_usage=True,
            )
        else:
            from diffusers import DiffusionPipeline

            pipe = DiffusionPipeline.from_pretrained(
                str(input_path),
                torch_dtype=target_dtype,
                low_cpu_mem_usage=True,
            )

        # Print converted model information and save
        if pipe is not None:
            print_pipeline_info(pipe, model_type, is_input=False)
            print(f"Saving in {target_dtype} as single file per component...")
            # Use very large max_shard_size to ensure single file output
            pipe.save_pretrained(str(output_path), max_shard_size="100GB")
    else:
        # For float8 and other non-standard dtypes, convert safetensors directly
        print("Float8 conversion - converting safetensors files directly...")
        convert_pipeline_safetensors(input_path, output_path, target_dtype)


def convert_single_model(
    input_path: Path, output_path: Path, target_dtype: torch.dtype = torch.bfloat16
):
    """Convert a single model component to target dtype."""
    import json

    with open(input_path / "config.json") as f:
        config = json.load(f)

    model_class = config.get("_class_name", "Unknown")
    print(f"Detected model: {model_class}")

    # First load without dtype conversion to see original dtypes
    print("\n>>> Loading input model to analyze original dtypes...")
    input_model = None

    if "Transformer" in model_class:
        from diffusers.models import Transformer2DModel

        input_model = Transformer2DModel.from_pretrained(
            str(input_path),
            low_cpu_mem_usage=True,
        )
    elif "UNet" in model_class:
        from diffusers import UNet2DConditionModel

        input_model = UNet2DConditionModel.from_pretrained(
            str(input_path),
            low_cpu_mem_usage=True,
        )
    elif "VAE" in model_class or "AutoencoderKL" in model_class:
        from diffusers import AutoencoderKL

        input_model = AutoencoderKL.from_pretrained(
            str(input_path),
            low_cpu_mem_usage=True,
        )
    else:
        print(
            f"Warning: Unknown model class '{model_class}', attempting safetensors conversion"
        )
        convert_safetensors_files(input_path, output_path, target_dtype)
        return

    # Print input model information
    if input_model is not None:
        print_model_info(input_model, model_class, is_input=True)
        del input_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Now convert to target dtype
    print(f"\n>>> Converting to {target_dtype}...")

    # For bf16, we can use torch_dtype parameter directly
    # For float8 types, convert safetensors directly
    if target_dtype == torch.bfloat16:
        model = None

        if "Transformer" in model_class:
            from diffusers.models import Transformer2DModel

            model = Transformer2DModel.from_pretrained(
                str(input_path),
                torch_dtype=target_dtype,
                low_cpu_mem_usage=True,
            )
        elif "UNet" in model_class:
            from diffusers import UNet2DConditionModel

            model = UNet2DConditionModel.from_pretrained(
                str(input_path),
                torch_dtype=target_dtype,
                low_cpu_mem_usage=True,
            )
        elif "VAE" in model_class or "AutoencoderKL" in model_class:
            from diffusers import AutoencoderKL

            model = AutoencoderKL.from_pretrained(
                str(input_path),
                torch_dtype=target_dtype,
                low_cpu_mem_usage=True,
            )

        # Print converted model information and save
        if model is not None:
            print_model_info(model, model_class, is_input=False)
            print(f"Saving in {target_dtype} as single file...")
            # Use very large max_shard_size to ensure single file output
            model.save_pretrained(str(output_path), max_shard_size="100GB")
    else:
        # For float8 and other non-standard dtypes, convert safetensors directly
        print("Float8 conversion - converting safetensors files directly...")
        convert_safetensors_files(input_path, output_path, target_dtype)


def convert_single_file(
    input_file: Path, output_path: Path, target_dtype: torch.dtype = torch.bfloat16
):
    """Convert a single safetensors or bin file to target dtype."""
    from collections import Counter

    from safetensors.torch import load_file, save_file

    # Determine output file path
    if output_path.suffix in [".safetensors", ".bin"]:
        # output_path is a file
        output_file = output_path
        output_dir = output_path.parent
    else:
        # output_path is a directory
        output_dir = output_path
        output_file = output_dir / input_file.name
        # Change extension to .safetensors for bin files
        if output_file.suffix == ".bin":
            output_file = output_file.with_suffix(".safetensors")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_file.name}...")

    # Load tensors
    if input_file.suffix == ".safetensors":
        tensors = load_file(input_file)
    else:  # .bin
        tensors = torch.load(input_file, map_location="cpu", weights_only=True)

    # Count input dtypes
    input_dtypes: Counter[str] = Counter()
    output_dtypes: Counter[str] = Counter()
    converted_tensors: dict[str, torch.Tensor] = {}

    for k, v in tensors.items():
        if isinstance(v, torch.Tensor):
            input_dtypes[str(v.dtype)] += v.numel()
            converted = v.to(target_dtype) if v.is_floating_point() else v
            converted_tensors[k] = converted
            output_dtypes[str(converted.dtype)] += converted.numel()

    # Print dtype summary with size estimates
    print("\n[INPUT] Dtype Distribution:")
    total_elements = sum(input_dtypes.values())
    total_bytes = 0
    for dtype, count in sorted(input_dtypes.items(), key=lambda x: -x[1]):
        pct = count / total_elements * 100
        bytes_per_elem = _get_dtype_size(dtype)
        size_bytes = count * bytes_per_elem
        total_bytes += size_bytes
        print(f"  {dtype}: {count:,} elements ({pct:.1f}%) - {format_size(size_bytes)}")
    print(f"  Total estimated size: {format_size(total_bytes)}")

    print("\n[OUTPUT] Dtype Distribution:")
    total_elements = sum(output_dtypes.values())
    total_bytes = 0
    for dtype, count in sorted(output_dtypes.items(), key=lambda x: -x[1]):
        pct = count / total_elements * 100
        bytes_per_elem = _get_dtype_size(dtype)
        size_bytes = count * bytes_per_elem
        total_bytes += size_bytes
        print(f"  {dtype}: {count:,} elements ({pct:.1f}%) - {format_size(size_bytes)}")
    print(f"  Total estimated size: {format_size(total_bytes)}")

    # Save converted tensors
    print(f"\nSaving {len(converted_tensors)} tensors to: {output_file.name}")
    save_file(converted_tensors, output_file)


def convert_safetensors_files(
    input_path: Path, output_path: Path, target_dtype: torch.dtype = torch.bfloat16
):
    """Convert safetensors files directly to target dtype and merge into a single file."""
    from collections import Counter

    from safetensors.torch import load_file, save_file

    output_path.mkdir(parents=True, exist_ok=True)

    # Copy non-tensor files
    for f in input_path.iterdir():
        if f.suffix not in [".safetensors", ".bin"]:
            if f.is_file():
                shutil.copy2(f, output_path / f.name)

    # Collect all tensors from all files and convert
    all_tensors: dict[str, torch.Tensor] = {}
    all_input_dtypes: Counter[str] = Counter()
    all_output_dtypes: Counter[str] = Counter()

    # Load from safetensors files
    for f in sorted(input_path.glob("*.safetensors")):
        print(f"Loading {f.name}...")
        tensors = load_file(f)

        # Count input dtypes
        for v in tensors.values():
            all_input_dtypes[str(v.dtype)] += v.numel()

        # Convert and add to collection
        for k, v in tensors.items():
            converted = v.to(target_dtype) if v.is_floating_point() else v
            all_tensors[k] = converted
            all_output_dtypes[str(converted.dtype)] += converted.numel()

    # Load from bin files
    for f in sorted(input_path.glob("*.bin")):
        print(f"Loading {f.name}...")
        tensors = torch.load(f, map_location="cpu", weights_only=True)

        # Count input dtypes and convert
        for k, v in tensors.items():
            if isinstance(v, torch.Tensor):
                all_input_dtypes[str(v.dtype)] += v.numel()
                converted = v.to(target_dtype) if v.is_floating_point() else v
                all_tensors[k] = converted
                all_output_dtypes[str(converted.dtype)] += converted.numel()

    # Print dtype summary with size estimates
    if all_input_dtypes:
        print("\n[INPUT] Dtype Distribution:")
        total_elements = sum(all_input_dtypes.values())
        total_bytes = 0
        for dtype, count in sorted(all_input_dtypes.items(), key=lambda x: -x[1]):
            pct = count / total_elements * 100
            bytes_per_elem = _get_dtype_size(dtype)
            size_bytes = count * bytes_per_elem
            total_bytes += size_bytes
            print(
                f"  {dtype}: {count:,} elements ({pct:.1f}%) - {format_size(size_bytes)}"
            )
        print(f"  Total estimated size: {format_size(total_bytes)}")

    if all_output_dtypes:
        print("\n[OUTPUT] Dtype Distribution:")
        total_elements = sum(all_output_dtypes.values())
        total_bytes = 0
        for dtype, count in sorted(all_output_dtypes.items(), key=lambda x: -x[1]):
            pct = count / total_elements * 100
            bytes_per_elem = _get_dtype_size(dtype)
            size_bytes = count * bytes_per_elem
            total_bytes += size_bytes
            print(
                f"  {dtype}: {count:,} elements ({pct:.1f}%) - {format_size(size_bytes)}"
            )
        print(f"  Total estimated size: {format_size(total_bytes)}")

    # Save all tensors to a single file
    if all_tensors:
        output_file = output_path / "model.safetensors"
        print(f"\nSaving {len(all_tensors)} tensors to single file: {output_file.name}")
        save_file(all_tensors, output_file)


def convert_pipeline_safetensors(
    input_path: Path, output_path: Path, target_dtype: torch.dtype
):
    """Convert a pipeline's safetensors files directly to target dtype."""

    output_path.mkdir(parents=True, exist_ok=True)

    # Copy model_index.json and other config files
    for f in input_path.iterdir():
        if f.is_file() and f.suffix in [".json", ".txt"]:
            shutil.copy2(f, output_path / f.name)

    # Process each subdirectory (transformer, vae, text_encoder, etc.)
    for subdir in input_path.iterdir():
        if subdir.is_dir():
            output_subdir = output_path / subdir.name
            print(f"\nProcessing {subdir.name}...")

            # Check if it has safetensors files
            safetensors_files = list(subdir.glob("*.safetensors")) + list(
                subdir.glob("*.bin")
            )
            if safetensors_files:
                convert_safetensors_files(subdir, output_subdir, target_dtype)
            else:
                # Just copy the directory (e.g., tokenizer, scheduler)
                if not output_subdir.exists():
                    shutil.copytree(subdir, output_subdir)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a diffusers model to reduced precision (bf16 or float8)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert to bfloat16 (default)
    python scripts/convert_floats.py --input H:/my-model
    # Output will be H:/my-model-bf16

    # Convert to float8 e4m3fn (higher precision)
    python scripts/convert_floats.py --input H:/my-model --dtype e4m3fn
    # Output will be H:/my-model-e4m3fn

    # Convert to float8 e5m2 (wider range)
    python scripts/convert_floats.py --input H:/my-model --dtype e5m2
    # Output will be H:/my-model-e5m2

    # Convert a single safetensors file
    python scripts/convert_floats.py --input H:/models/model.safetensors
    # Output will be H:/models/model-bf16.safetensors

    # Convert to a specific output path
    python scripts/convert_floats.py --input H:/my-model --output H:/converted/my-model
        """,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input model directory or safetensors/bin file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the converted model (default: input path + dtype suffix)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "e4m3fn", "e5m2"],
        default="bf16",
        help="Target dtype: bf16 (bfloat16), e4m3fn (float8, higher precision), or e5m2 (float8, wider range). Default: bf16",
    )

    args = parser.parse_args()

    convert_model(args.input, args.output, args.dtype)


if __name__ == "__main__":
    main()
