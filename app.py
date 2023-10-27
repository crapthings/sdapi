import runpod
import torch
from diffusers import StableDiffusionPipeline, AutoPipelineForImage2Image
from diffusers.pipelines.stable_diffusion import safety_checker
from diffusers.utils import load_image
from PIL import Image
import numpy as np

from utils import rounded_size, sc, getSampler, upload_file

# override
safety_checker.StableDiffusionSafetyChecker.forward = sc

model1 = './dreamshaper_8.safetensors'

txt2imgPipe = StableDiffusionPipeline.from_single_file(model1, torch_dtype = torch.float16).to('cuda')
txt2imgPipe.scheduler = getSampler('EulerAncestralDiscreteScheduler', txt2imgPipe.scheduler.config)
txt2imgPipe.enable_xformers_memory_efficient_attention()

img2imgPipe = AutoPipelineForImage2Image.from_pipe(txt2imgPipe).to('cuda')
img2imgPipe.scheduler = getSampler('EulerAncestralDiscreteScheduler', img2imgPipe.scheduler.config)
img2imgPipe.enable_xformers_memory_efficient_attention()

def render (job, _generator = None, _output = None):
    _id = job.get('id')
    _input = job.get('input')

    print('debug', job, _input)

    image_url = _input.get('image_url', None)
    prompt = _input.get('prompt', 'a dog')
    height = _input.get('height', 512)
    width = _input.get('width', 512)
    num_inference_steps = int(np.clip(_input.get('num_inference_steps', 30), 10, 150))
    guidance_scale = float(np.clip(_input.get('guidance_scale', 13), 0, 30))
    negative_prompt = _input.get('negative_prompt', None)

    sampler = _input.get('sampler', 'EulerAncestralDiscreteScheduler')
    seed = _input.get('seed', None)
    hires = _input.get('hires', None)
    strength = float(np.clip(_input.get('strength', .5), 0, 1))
    scale = float(np.clip(_input.get('scale', 2), 1, 2))

    _debug = _input.get('debug', False)

    roundedWidth, roundedHeight = rounded_size(width, height)

    props = {
        prompt: prompt,
        negative_prompt: negative_prompt,
        image: _output,
        num_inference_steps: num_inference_steps,
        guidance_scale: guidance_scale,
        strength: strength,
        generator: _generator,
    }

    if seed is not None:
        _generator = torch.Generator(device = 'cuda').manual_seed(seed)

    if image_url is not None:
        _output = load_image(image_url)
        img2imgPipe.scheduler = getSampler(sampler, img2imgPipe.scheduler.config)
        _output = img2imgPipe(**props).images[0]
    else:
        txt2imgPipe.scheduler = getSampler(sampler, txt2imgPipe.scheduler.config)
        _output = txt2imgPipe(**props).images[0]

    if hires:
        _output = _output.resize([int(width * scale), int(height * scale)])
        _output = img2imgPipe(**props).images[0]

    _output = _output.resize([width, height])

    filename = upload_file(_output)

    if _debug:
        _output.save('./debug.png')

    result = {
        '_job_id': _id,
        'filename': filename,
        'prompt': prompt,
        'height': height,
        'width': width,
        'num_inference_steps': num_inference_steps,
        'guidance_scale': guidance_scale,
        'negative_prompt': negative_prompt,
        'sampler': sampler,
        'seed': seed,
        'hires': hires,
        'strength': strength,
        'scale': scale
    }

    return result

runpod.serverless.start({ 'handler': render })
