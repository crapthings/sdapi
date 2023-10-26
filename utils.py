import os
import io
import base64
import uuid
import logging
import boto3
from botocore.exceptions import ClientError
from diffusers import EulerAncestralDiscreteScheduler, UniPCMultistepScheduler

aws_access_key_id = os.environ.get('AWS_KEY')
aws_secret_access_key = os.environ.get('AWS_SECRET')
bucket_name = os.environ.get('AWS_BUCKET')

def upload_file (file, object_name = None):
    s3_client = boto3.client('s3', aws_access_key_id = aws_access_key_id, aws_secret_access_key = aws_secret_access_key)

    buffer = io.BytesIO()

    file.save(buffer, 'PNG')

    buffer.seek(0)

    if object_name is None:
        object_name = str(uuid.uuid4()) + '.png'

    try:
        response = s3_client.upload_fileobj(buffer, bucket_name, object_name)
    except ClientError as e:
        logging.error(e)
        return None

    return object_name

#
def rounded_size (width, height):
    rounded_width = (width // 8) * 8
    rounded_height = (height // 8) * 8

    if width % 8 >= 4:
        rounded_width += 8
    if height % 8 >= 4:
        rounded_height += 8

    return int(rounded_width), int(rounded_height)

# override
def sc(self, clip_input, images): return images, [False for i in images]

#
def getSampler (name, config):
    sampler = {
        'EulerAncestralDiscreteScheduler': EulerAncestralDiscreteScheduler.from_config(config),
        'UniPCMultistepScheduler': UniPCMultistepScheduler.from_config(config),
    }

    return sampler.get(name)

def encodeBase64Img (image):
    buffer = io.BytesIO()
    image.save(buffer, 'PNG')
    buffer.seek(0)
    image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return image
