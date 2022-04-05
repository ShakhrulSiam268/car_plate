import cv2
import time
import easyocr
from custom_utils import *
reader = easyocr.Reader(['bn'])

source = './test/car4.jpg'
model_path = 'detection_best.pt'

img0 = cv2.imread(source)
detected, cropped = decode_image(img0, model_path)
small_frame = cv2.resize(detected, (0, 0), fx=0.75, fy=0.75)

result = reader.readtext(cropped)

for res in result:
    text = res[1]
    print(text)

show_image(cropped)

