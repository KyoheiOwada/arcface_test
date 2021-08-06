import cv2
from PIL import Image

def cv2pil(image_cv)
  image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
  image_pil = Image.fromarray(image_cv)
  image_pil = image_pil.convert('RGB')

  return image_pil
