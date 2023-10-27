import runpod
import torch
from diffusers import StableDiffusionPipeline, AutoPipelineForImage2Image, StableDiffusionUpscalePipeline
from diffusers.pipelines.stable_diffusion import safety_checker
from diffusers.utils import load_image
from PIL import Image
import numpy as np

from utils import rounded_size, sc, getSampler, upload_file

# override
safety_checker.StableDiffusionSafetyChecker.forward = sc

model1 = './dreamshaper_8.safetensors'

txt2imgPipe = StableDiffusionPipeline.from_single_file(model1, torch_dtype = torch.float16, safety_checker = None).to('cuda')
txt2imgPipe.scheduler = getSampler('EulerAncestralDiscreteScheduler', txt2imgPipe.scheduler.config)
txt2imgPipe.enable_xformers_memory_efficient_attention()

img2imgPipe = AutoPipelineForImage2Image.from_pipe(txt2imgPipe).to('cuda')
img2imgPipe.scheduler = getSampler('EulerAncestralDiscreteScheduler', img2imgPipe.scheduler.config)
img2imgPipe.enable_xformers_memory_efficient_attention()

sdupscalerPipe = StableDiffusionUpscalePipeline.from_pretrained('stabilityai/stable-diffusion-x4-upscaler', revision = 'fp16', torch_dtype = torch.float16).to('cuda')
sdupscalerPipe.enable_xformers_memory_efficient_attention()

def render (job, _output = None):
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

    enable_2pass = _input.get('enable_2pass', None)
    strength = float(np.clip(_input.get('strength', .5), 0, 1))
    scale = float(np.clip(_input.get('scale', 2), 1, 2))

    upscale_url = _input.get('upscale_url', None)

    _debug = _input.get('debug', False)

    roundedWidth, roundedHeight = rounded_size(width, height)

    props = {
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'num_inference_steps': num_inference_steps,
        'guidance_scale': guidance_scale,
    }

    if upscale_url is not None:
        _output = sdupscalerPipe(image = load_image(upscale_url), **props).images[0]
        filename = upload_file(_output)

        return {
            '_job_id': _id,
            'prompt': prompt,
            'height': _output.height,
            'width': _output.width,
            'upscale_url': upscale_url,
            'filename': filename,
            'negative_prompt': negative_prompt,
        }

    if seed is not None:
        props['generator'] = torch.Generator(device = 'cuda').manual_seed(seed)

    if image_url is not None:
        image = load_image(image_url)
        img2imgPipe.scheduler = getSampler(sampler, img2imgPipe.scheduler.config)
        _output = img2imgPipe(image = image, strength = strength, **props).images[0]

        if enable_2pass:
            _output = _output.resize([int(_output.width * scale), int(_output.height * scale)], Image.Resampling.LANCZOS)
            _output = img2imgPipe(image = _output, **props).images[0]
            _output = _output.resize([image.width, image.height], Image.Resampling.LANCZOS)

    else:
        txt2imgPipe.scheduler = getSampler(sampler, txt2imgPipe.scheduler.config)
        _output = txt2imgPipe(height = roundedHeight, width = roundedWidth, **props).images[0]

        if enable_2pass:
            _output = _output.resize([int(_output.width * scale), int(_output.height * scale)], Image.Resampling.LANCZOS)
            _output = img2imgPipe(image = _output, strength = strength, **props).images[0]
            _output = _output.resize([width, height], Image.Resampling.LANCZOS)

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
        'enable_2pass': enable_2pass,
        'strength': strength,
        'scale': scale
    }

    return result

runpod.serverless.start({ 'handler': render })
