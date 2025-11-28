import click
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


@click.command()
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
def main(base_model, lora_path, prompt, output, steps):
    """
    Generate an image using a trained LoRA.
    """
    print(f"Loading base model: {base_model}")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16, safety_checker=None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    print(f"Loading LoRA from: {lora_path}")
    # If lora_path is a directory, PEFT usually saves as adapter_model.bin inside
    # pipe.load_lora_weights handles both directory and file usually
    pipe.load_lora_weights(lora_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)

    print(f"Generating: '{prompt}'")
    image = pipe(prompt, num_inference_steps=steps).images[0]

    image.save(output)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
