import runpod
import torch
from diffusers import StableDiffusionPipeline

model = 'runwayml/stable-diffusion-v1-5'

pipe = StableDiffusionPipeline.from_pretrained(
    model,
    cache_dir = './',
    # torch_dtype = torch.float16,
    use_safetensors = True,
)

def is_even (job):
    print(job)

    # prompt = 'a photo of an astronaut riding a horse on mars'

    # image = pipe(prompt).images[0]

    return 1

runpod.serverless.start({ 'handler': is_even })
