from typing import Optional
import torch


def generate_zimage(
    base_model: str,
    lora_path: str,
    prompt: str,
    output: str = "output.png",
    width: int = 1024,
    height: int = 1024,
    steps: int = 8,
    seed: Optional[int] = None,
    lora_scale: float = 1.0,
):
    """
    Generate an image using a trained Z-Image-Turbo LoRA.

    Z-Image-Turbo is optimized for 8 inference steps and produces high-quality
    photorealistic images with excellent text rendering in English and Chinese.

    Args:
        base_model: Z-Image model ID (default: Tongyi-MAI/Z-Image-Turbo)
        lora_path: Path to trained LoRA directory
        prompt: Prompt to generate
        output: Output filename
        width: Image width (default: 1024)
        height: Image height (default: 1024)
        steps: Inference steps (default: 8 for turbo)
        seed: Random seed (optional)
        lora_scale: LoRA scale (default: 1.0)
    """
    try:
        from diffusers import ZImagePipeline
    except ImportError:
        raise ImportError(
            "Z-Image support requires the latest diffusers from source.\n"
            "Install with: pip install git+https://github.com/huggingface/diffusers"
        )

    print(f"Loading Z-Image model: {base_model}")
    pipe = ZImagePipeline.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
    )

    print(f"Loading LoRA from: {lora_path}")
    # Load the trained LoRA adapter
    pipe.load_lora_weights(lora_path)

    # Set LoRA scale if not 1.0
    if lora_scale != 1.0:
        pipe.set_adapters(["default"], adapter_weights=[lora_scale])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)

    # Enable memory optimizations
    pipe.enable_model_cpu_offload()

    print(f"Generating: '{prompt}'")
    print(f"Resolution: {width}x{height}, Steps: {steps}")

    generator = None
    if seed is not None:
        generator = torch.Generator(device).manual_seed(seed)
        print(f"Using seed: {seed}")

    # Z-Image-Turbo uses guidance_scale=0.0
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps + 1,  # +1 because Z-Image counts differently
        guidance_scale=0.0,  # Turbo models don't use guidance
        generator=generator,
    ).images[0]

    image.save(output)
    print(f"Saved to {output}")
    return output
