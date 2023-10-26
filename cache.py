import torch
from diffusers import StableDiffusionPipeline, AutoPipelineForImage2Image

model1 = './dreamshaper_8.safetensors'
txt2imgPipe = StableDiffusionPipeline.from_single_file(model1, torch_dtype = torch.float16)
AutoPipelineForImage2Image.from_pipe(txt2imgPipe)
