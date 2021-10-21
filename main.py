import cv2
import time
import easyocr
from custom_utils import *

# C:\Users\User\AppData\Roaming\Python\Python38\site-packages\easyocr\dict


source='./test/car4.jpg'
model_path='detection_best.pt'

img0= cv2.imread(source)
detected,cropped=decode_image(img0,model_path)
small_frame = cv2.resize(detected, (0, 0), fx=0.5, fy=0.5)

#show_image(small_frame)

reader = easyocr.Reader(['bn'])
result = reader.readtext(cropped,paragraph="False")
try:
    text=result[0][1]
except:
    text='unable to detect'
print(text)
show_image(cropped)

