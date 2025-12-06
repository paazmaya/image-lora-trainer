import click
from pathlib import Path


@click.group()
def cli():
    """Image LoRA Trainer - Train LoRA adapters for various diffusion models."""
    pass


@cli.command("train")
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to input images",
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(),
    help="Output directory (default: cwd)",
)
@click.option(
    "--base-model",
    default="runwayml/stable-diffusion-v1-5",
    help="Base model ID or path",
)
@click.option("--resolution", default=512, help="Training resolution")
@click.option(
    "--instance-prompt", default="a photo of a sks person", help="Instance prompt"
)
@click.option(
    "--crop-focus", default=None, help="Object to focus crop on (e.g. 'face', 'person')"
)
@click.option("--use-qlora", is_flag=True, help="Use QLoRA (4-bit quantization)")
@click.option("--epochs", default=None, type=int, help="Number of epochs")
@click.option("--steps", default=1000, type=int, help="Number of training steps")
def train_sd(
    input_dir,
    output_dir,
    base_model,
    resolution,
    instance_prompt,
    crop_focus,
    use_qlora,
    epochs,
    steps,
):
    """
    Train a LoRA adapter for Stable Diffusion / SDXL from a folder of images.
    """
    from stable_diffusion.train import train_lora

    input_path = Path(input_dir)

    if output_dir is None:
        # Default to cwd with constructed name
        base_name = (
            Path(base_model).stem
            if Path(base_model).exists()
            else base_model.split("/")[-1]
        )
        folder_name = input_path.name
        output_name = f"{base_name}_{folder_name}"
        final_output_dir = Path.cwd() / output_name
    else:
        # Use the user-specified output directory as-is
        final_output_dir = Path(output_dir)

    final_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training LoRA for {base_model} on {input_dir}")
    print(f"Output will be saved to {final_output_dir}")

    train_lora(
        input_dir=str(input_path),
        output_dir=str(final_output_dir),
        base_model=base_model,
        resolution=resolution,
        instance_prompt=instance_prompt,
        crop_focus=crop_focus,
        use_qlora=use_qlora,
        num_train_epochs=epochs,
        max_train_steps=steps,
    )


@cli.command("train-zimage")
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to input images",
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(),
    help="Output directory (default: cwd)",
)
@click.option(
    "--base-model",
    default="Tongyi-MAI/Z-Image-Turbo",
    help="Z-Image model ID (default: Tongyi-MAI/Z-Image-Turbo)",
)
@click.option("--resolution", default=1024, help="Training resolution (default: 1024)")
@click.option(
    "--instance-prompt", default="a photo of a sks person", help="Instance prompt"
)
@click.option(
    "--crop-focus", default=None, help="Object to focus crop on (e.g. 'face', 'person')"
)
@click.option("--use-8bit", is_flag=True, help="Use 8-bit quantization for lower VRAM")
@click.option(
    "--no-training-adapter",
    is_flag=True,
    help="Disable the de-distillation training adapter (not recommended)",
)
@click.option("--epochs", default=None, type=int, help="Number of epochs")
@click.option("--steps", default=1000, type=int, help="Number of training steps")
@click.option("--lr", default=1e-5, type=float, help="Learning rate (default: 1e-5)")
@click.option("--lora-rank", default=16, type=int, help="LoRA rank (default: 16)")
@click.option("--lora-alpha", default=16, type=int, help="LoRA alpha (default: 16)")
@click.option(
    "--save-steps", default=500, type=int, help="Save checkpoint every N steps"
)
def train_zimage(
    input_dir,
    output_dir,
    base_model,
    resolution,
    instance_prompt,
    crop_focus,
    use_8bit,
    no_training_adapter,
    epochs,
    steps,
    lr,
    lora_rank,
    lora_alpha,
    save_steps,
):
    """
    Train a LoRA adapter for Z-Image-Turbo from a folder of images.

    Z-Image-Turbo is a fast 8-step diffusion transformer model. This command
    uses a special training adapter (ostris/zimage_turbo_training_adapter) to
    prevent the distillation from breaking during training.

    Example:
        uv run python main.py train-zimage --input-dir ./my_images --instance-prompt "photo of sks person"
    """
    from zimage.train import train_zimage_lora

    input_path = Path(input_dir)

    if output_dir is None:
        folder_name = input_path.name
        output_name = f"zimage-turbo_{folder_name}"
        final_output_dir = Path.cwd() / output_name
    else:
        final_output_dir = Path(output_dir)

    final_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training Z-Image-Turbo LoRA on {input_dir}")
    print(f"Output will be saved to {final_output_dir}")
    if use_8bit:
        print("Using 8-bit quantization for reduced VRAM usage")
    if not no_training_adapter:
        print("Using training adapter (ostris/zimage_turbo_training_adapter)")

    train_zimage_lora(
        input_dir=str(input_path),
        output_dir=str(final_output_dir),
        base_model=base_model,
        resolution=resolution,
        instance_prompt=instance_prompt,
        crop_focus=crop_focus,
        use_8bit=use_8bit,
        use_training_adapter=not no_training_adapter,
        num_train_epochs=epochs,
        max_train_steps=steps,
        learning_rate=lr,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        save_steps=save_steps,
    )


# Keep backward compatibility - default command is train
@cli.command("main", hidden=True)
@click.pass_context
def main_compat(ctx):
    """Backward compatibility entry point."""
    ctx.invoke(train_sd)


if __name__ == "__main__":
    cli()
