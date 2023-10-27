import torch
from diffusers import StableDiffusionPipeline, AutoPipelineForImage2Image, StableDiffusionUpscalePipeline

model1 = './dreamshaper_8.safetensors'
txt2imgPipe = StableDiffusionPipeline.from_single_file(model1, torch_dtype = torch.float16, safety_checker = None)
AutoPipelineForImage2Image.from_pipe(txt2imgPipe, safety_checker = None)
StableDiffusionUpscalePipeline.from_pretrained('stabilityai/stable-diffusion-x4-upscaler', revision = 'fp16', torch_dtype = torch.float16)
