import os
import time
from tqdm import tqdm, trange
from pprint import pprint
import numpy as np
import cv2
import torch
import torchvision as tv

from models import YOLOv3
from utils import COCODataset, draw_bboxes, bboxes_to_labels

# root = "data/COCO/val2017/"
# ann_file = "data/COCO/annotations/instances_val2017.json"
# coco = COCODataset(root, ann_file)
# for i in np.random.randint(0, len(coco), 25):
#   img, (bboxes, cats) = coco[i]
#   img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#   img = draw_bboxes(img, bboxes, cats)
#   cv2.imshow("Boxes", img)
#   cv2.waitKey(0)
# cv2.destroyAllWindows()

# root = "data/COCO/train2017/"
# ann_file = "data/COCO/annotations_trainval2017/annotations/instances_train2017.json"
# ds = tv.datasets.CocoDetection(root, ann_file)
# count = 0
# fnf = 0
# for i in trange(len(ds)):
#     try:
#         img, lbl = ds[i]
#         w, h = img.size
#         if w>640 or h>640:
#             count += 1
#     except:
#         fnf += 1
#         continue
# print("Result: ", count)
# print("Files not found: ", fnf)

root = "data/COCO/cleaned/val2017/"
coco = COCODataset(root)
for i in np.random.randint(0, len(coco), 25):
    img, (bboxes, cats) = coco[i]
    obj_mask, noobj_mask, label = bboxes_to_labels(bboxes, cats, 32)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = draw_bboxes(img, bboxes, cats)
    cv2.imshow("Boxes", img)
    cv2.waitKey(0)
cv2.destroyAllWindows()
