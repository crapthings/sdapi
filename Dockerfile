FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer='crapthings@gmail.com'

WORKDIR /workspace

ADD https://huggingface.co/digiplay/DreamShaper_8/resolve/main/dreamshaper_8.safetensors .
COPY app.py .
COPY utils.py .
COPY cache.py .

RUN pip install runpod \
  accelerate \
  xformers \
  transformers \
  diffusers['torch'] \
  omegaconf \
  boto3

RUN python ./cache.py

CMD python -u ./app.py
