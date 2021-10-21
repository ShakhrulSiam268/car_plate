import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox


def show_image(img):
    cv2.imshow('Detected_Frame',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def decode_image(img0,model_path):

    device='cpu'
    w=model_path
    model=attempt_load(w, map_location=device)
    stride = int(model.stride.max())  
    names = model.module.names if hasattr(model, 'module') else model.names  

    im0s=img0
    img = letterbox(img0, 640, 32)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()  
    img = img / 255.0  
    if len(img.shape) == 3:
        img = img[None]

    pred = model(img)[0]
    imgsz=640
    conf_thres=0.25 
    iou_thres=0.45  
    max_det=1000
    classes=None
    agnostic_nms=False
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    source=" "
    path=source
    det=pred[0]
    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

    im2=img0
    #crop_im=[]
    crop_im=img0
    for i in range (len(det)):
        left, top, right, bottom=np.array(det[i,0:4])

        try:
            crop_im=img0[int(top):int(bottom), int(left):int(right)]
        except:
            crop_im=img0

        index=int(np.array(det[i,5]))
        score=round(float(np.array(det[i,4])),2)
        #print(names[index],score)
        cv2.rectangle(im2, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
        #cv2.rectangle(im2, (int(left), int(top) + 35), (int(right), int(top)), (0, 255, 0), cv2.FILLED)
        cv2.rectangle(im2, (int(left), int(top) - 35), (int(left)+len( names[index])*25, int(top)), (0, 255, 0), cv2.FILLED)
        cv2.putText(im2, names[index], (int(left) + 20, int(top) -8), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 1)
        #crop_im.append(im2[y:y+h, x:x+w])


    return im2,crop_im

