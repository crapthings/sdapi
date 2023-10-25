python -m venv venv

source venv/bin/activate

pip install runpod \
  accelerate \
  xformers \
  transformers \
  diffusers['torch'] \
  omegaconf

python3 app.py
