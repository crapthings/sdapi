from PIL import Image
from utils import upload_file

img = Image.open('./sample.png')
filename = upload_file(img)
print(filename)
