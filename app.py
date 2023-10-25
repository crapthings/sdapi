import io
import base64
import runpod
import torch
from diffusers import StableDiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler, UniPCMultistepScheduler
from diffusers.pipelines.stable_diffusion import safety_checker
from PIL import Image
import numpy as np

# override
def sc(self, clip_input, images): return images, [False for i in images]
safety_checker.StableDiffusionSafetyChecker.forward = sc

def getSampler (name, config):
    sampler = {
        'EulerAncestralDiscreteScheduler': EulerAncestralDiscreteScheduler.from_config(config),
        'UniPCMultistepScheduler': UniPCMultistepScheduler.from_config(config),
    }

    return sampler.get(name)

model = 'Lykon/dreamshaper-8'

txt2imgPipe = StableDiffusionPipeline.from_pretrained(
    model,
    cache_dir = './',
    torch_dtype = torch.float16,
    use_safetensors = True,
)
txt2imgPipe.scheduler = getSampler('EulerAncestralDiscreteScheduler', txt2imgPipe.scheduler.config)
txt2imgPipe.enable_model_cpu_offload()
txt2imgPipe.enable_xformers_memory_efficient_attention()

def render (job, _generator = None):
    _id = job.get('id')
    _input = job.get('input')

    print('debug', job, _input)

    prompt = _input.get('prompt', 'a dog')
    height = _input.get('height', 512)
    width = _input.get('width', 512)
    num_inference_steps = np.clip(_input.get('num_inference_steps', 30), 10, 150)
    guidance_scale = np.clip(_input.get('guidance_scale', 13), 0, 30)
    negative_prompt = _input.get('negative_prompt', None)

    sampler = _input.get('sampler', 'EulerAncestralDiscreteScheduler')
    seed = _input.get('seed', None)

    _webhook = _input.get('webhook', None)
    _debug = _input.get('debug', False)

    roundedWidth, roundedHeight = rounded_size(width, height)

    txt2imgPipe.scheduler = getSampler(sampler, txt2imgPipe.scheduler.config)

    if seed is not None:
        _generator = torch.Generator(device = 'cuda').manual_seed(seed)

    output = txt2imgPipe(
        prompt,
        height = roundedHeight,
        width = roundedWidth,
        num_inference_steps = num_inference_steps,
        guidance_scale = guidance_scale,
        negative_prompt = negative_prompt,
        generator = _generator,
    ).images[0]

    output = output.resize([width, height])

    buffer = io.BytesIO()

    output.save(buffer, 'PNG')

    buffer.seek(0)

    output_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    if _debug:
        output.save('./debug.png')

    result = {
        '_job_id': _id,
        'output_image': 1 or output_image,
        'prompt': prompt,
        'height': height,
        'width': width,
        'num_inference_steps': num_inference_steps,
        'guidance_scale': guidance_scale,
        'negative_prompt': negative_prompt,
        'sampler': sampler,
        'seed': seed,
        'webhook': _webhook
    }

    return result

def rounded_size (width, height):
    rounded_width = (width // 8) * 8
    rounded_height = (height // 8) * 8

    if width % 8 >= 4:
        rounded_width += 8
    if height % 8 >= 4:
        rounded_height += 8

    return int(rounded_width), int(rounded_height)

runpod.serverless.start({ 'handler': render })
