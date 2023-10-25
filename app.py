import io
import base64
import runpod
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# model = 'runwayml/stable-diffusion-v1-5'

# pipe = StableDiffusionPipeline.from_pretrained(
#     model,
#     cache_dir = './',
#     # torch_dtype = torch.float16,
#     use_safetensors = True,
# )

def render (job):
    _input = job.get('input')

    # output = pipe(
    #     prompt,
    #     num_inference_steps = 20,
    # ).images[0]

    output = Image.open('./sample.png')

    buffer = io.BytesIO()

    output.save(buffer, 'PNG')

    buffer.seek(0)

    image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return {
        'output_image': 1,
    }

runpod.serverless.start({ 'handler': render })
