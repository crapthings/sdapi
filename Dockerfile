FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer='crapthings@gmail.com'

WORKDIR /workspace

COPY app.py .
COPY utils.py .

RUN apt-get update && pip install runpod \
  accelerate \
  xformers \
  transformers \
  diffusers['torch'] \
  omegaconf

CMD ['python', '-u', '/app.py']
