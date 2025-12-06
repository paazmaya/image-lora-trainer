import click


@click.group()
def cli():
    """Generate images using trained LoRA adapters."""
    pass


@cli.command("sd")
@click.option(
    "--base-model",
    default="runwayml/stable-diffusion-v1-5",
    help="Base model ID or path",
)
@click.option(
    "--lora-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to LoRA directory or file",
)
@click.option("--prompt", required=True, help="Prompt to generate")
@click.option("--output", default="output.png", help="Output filename")
@click.option("--steps", default=30, help="Inference steps")
def generate_sd_cmd(base_model, lora_path, prompt, output, steps):
    """
    Generate an image using a trained Stable Diffusion LoRA.
    """
    from stable_diffusion.generate import generate_sd

    generate_sd(
        base_model=base_model,
        lora_path=lora_path,
        prompt=prompt,
        output=output,
        steps=steps,
    )


@cli.command("zimage")
@click.option(
    "--base-model",
    default="Tongyi-MAI/Z-Image-Turbo",
    help="Z-Image model ID (default: Tongyi-MAI/Z-Image-Turbo)",
)
@click.option(
    "--lora-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to trained LoRA directory",
)
@click.option("--prompt", required=True, help="Prompt to generate")
@click.option("--output", default="output.png", help="Output filename")
@click.option("--width", default=1024, type=int, help="Image width (default: 1024)")
@click.option("--height", default=1024, type=int, help="Image height (default: 1024)")
@click.option(
    "--steps", default=8, type=int, help="Inference steps (default: 8 for turbo)"
)
@click.option("--seed", default=None, type=int, help="Random seed (optional)")
@click.option("--lora-scale", default=1.0, type=float, help="LoRA scale (default: 1.0)")
def generate_zimage_cmd(
    base_model, lora_path, prompt, output, width, height, steps, seed, lora_scale
):
    """
    Generate an image using a trained Z-Image-Turbo LoRA.

    Z-Image-Turbo is optimized for 8 inference steps and produces high-quality
    photorealistic images with excellent text rendering in English and Chinese.

    Example:
        uv run python generate.py zimage --lora-path ./my_lora --prompt "photo of sks person"
    """
    from zimage.generate import generate_zimage

    generate_zimage(
        base_model=base_model,
        lora_path=lora_path,
        prompt=prompt,
        output=output,
        width=width,
        height=height,
        steps=steps,
        seed=seed,
        lora_scale=lora_scale,
    )


# Keep backward compatibility
@cli.command("main", hidden=True)
@click.pass_context
def main_compat(ctx):
    """Backward compatibility."""
    ctx.invoke(generate_sd_cmd)


if __name__ == "__main__":
    cli()
