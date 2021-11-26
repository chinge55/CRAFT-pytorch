import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT

from collections import OrderedDict


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


result_folder = './custom_result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

############################VARIABLES##############################
trained_model = 'craft_mlt_25k.pth'
image_path = 'data/2.PNG'
poly = False
canvas_size = 1280
text_threshold = 0.4
low_text = 0.4
link_threshold = 0.4
mag_ratio = 1.5






###################################################################
#Custom test for a single image
net = CRAFT()
net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu')))
net.eval()
image = imgproc.loadImage(image_path)
#############################################


img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
ratio_h = ratio_w = 1 / target_ratio

#Preprocessing
x = imgproc.normalizeMeanVariance(img_resized)
x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]


with torch.no_grad():
    y, feature = net(x)


score_text = y[0,:,:,0].cpu().data.numpy()
score_link = y[0,:,:,1].cpu().data.numpy()

    
#Post-Processing
boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

# coordinate adjustment
boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
for k in range(len(polys)):
    if polys[k] is None: polys[k] = boxes[k]

#Rendering the results
render_img = score_text.copy()
render_img = np.hstack((render_img, score_link))
score_text = imgproc.cvt2HeatmapImg(render_img)


# save score text
filename, file_ext = os.path.splitext(os.path.basename(image_path))
mask_file = result_folder + "/res_" + filename + '_mask.jpg'
cv2.imwrite(mask_file, score_text)
# cv2.imshow("image", score_text)
# cv2.waitKey(0)
file_utils.saveResult(image_path, image[:,:,::-1],polys, dirname=result_folder)