python -m venv venv

source venv/bin/activate

pip install runpod \
  accelerate \
  xformers \
  transformers \
  diffusers['torch'] \
  omegaconf \
  boto3

python app.py
