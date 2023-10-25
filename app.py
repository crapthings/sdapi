import runpod
import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import safety_checker
from PIL import Image
import numpy as np

from utils import rounded_size, sc, getSampler, encodeBase64Img

# override
safety_checker.StableDiffusionSafetyChecker.forward = sc

model = './dreamshaper_8.safetensors'

txt2imgPipe = StableDiffusionPipeline.from_single_file(
    model,
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

    output_image = encodeBase64Img(output)

    if _debug:
        output.save('./debug.png')

    result = {
        '_job_id': _id,
        'output_image': output_image,
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

runpod.serverless.start({ 'handler': render })
