import io
import base64
from diffusers import EulerAncestralDiscreteScheduler, UniPCMultistepScheduler

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
