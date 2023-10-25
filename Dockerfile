FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer='crapthings@gmail.com'

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

COPY app.py .
COPY utils.py .

RUN apt-get update -y && \
apt-get upgrade -y && \
apt-get install --yes --no-install-recommends \
build-essential \
vim \
git \
wget \
software-properties-common \
google-perftools \
curl \
bash

RUN apt-get autoremove -y && \
apt-get clean -y && \
rm -rf /var/lib/apt/lists/* && \
add-apt-repository ppa:deadsnakes/ppa -y && \
apt-get install python3.10 -y --no-install-recommends && \
ln -s /usr/bin/python3.10 /usr/bin/python && \
rm /usr/bin/python3 && \
ln -s /usr/bin/python3.10 /usr/bin/python3 && \
which python && \
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
python get-pip.py && \
rm get-pip.py && \
pip install -U pip

RUN pip install runpod \
  accelerate \
  xformers \
  transformers \
  diffusers['torch'] \
  omegaconf

CMD ['python', '-u', '/app.py']
