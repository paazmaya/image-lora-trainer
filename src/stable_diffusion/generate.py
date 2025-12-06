import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


def generate_sd(
    base_model: str,
    lora_path: str,
    prompt: str,
    output: str = "output.png",
    steps: int = 30,
):
    """
    Generate an image using a trained Stable Diffusion LoRA.
    """
    print(f"Loading base model: {base_model}")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model, dtype=torch.float16, safety_checker=None
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
    return output
