import click
from pathlib import Path
from train import train_lora


@click.command()
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
def main(
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
    Train a LoRA adapter from a folder of images.
    """
    input_path = Path(input_dir)

    if output_dir is None:
        # Default to cwd with constructed name
        base_name = Path(base_model).stem if Path(base_model).exists() else base_model.split("/")[-1]
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


if __name__ == "__main__":
    main()
