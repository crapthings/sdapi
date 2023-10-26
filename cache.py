import torch
from diffusers import StableDiffusionPipeline

model1 = './dreamshaper_8.safetensors'
StableDiffusionPipeline.from_single_file(model1, torch_dtype = torch.float16)
