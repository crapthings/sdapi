FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer='crapthings@gmail.com'

WORKDIR /workspace

COPY app.py .
COPY utils.py .
ADD https://huggingface.co/digiplay/DreamShaper_8/resolve/main/dreamshaper_8.safetensors .

RUN apt-get update && pip install runpod \
  accelerate \
  xformers \
  transformers \
  diffusers['torch']

CMD ['python', '-u', '/app.py']
