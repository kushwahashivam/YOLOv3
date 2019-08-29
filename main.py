import os
import time
from tqdm import tqdm, trange
from pprint import pprint
import numpy as np
import cv2
import torch
import torchvision as tv

from models import YOLOv3
from utils import COCODataset, draw_bboxes

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = tv.models.resnet34().to(device).eval()
# x = torch.randn(1, 3, 640, 480).to(device)
# with torch.no_grad():
#     for _ in trange(100):
#         _ = model(x)

# children = []
# for i in range(1, len(list(model.children()))-1):
#   children.append(torch.nn.Sequential(*list(model.children())[:i]).to(device))

# print("Input shape: ", x.shape)
# for i in range(len(children)):
#   y = children[i](x)
#   print("Output shape at ", i+1, " layer: ", y.shape)

# print("\nOutput size here: ", torch.nn.Sequential(*list(model.children())[:6]).to(device)(x).shape)

# yolo = YOLOv3().to(device).eval()
# inp = torch.randn(1, 3, 480, 640).to(device)
# pred = yolo(inp)
# print(pred.shape)
# with torch.no_grad():
#   for _ in trange(100):
#     pred = yolo(inp)

# normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# root = "data/COCO/val2017/"
# ann_file = "data/COCO/annotations/instances_val2017.json"
# ds = tv.datasets.CocoDetection(root, ann_file)
# img, lbl = ds[101]
# pprint(len(ds))
# pprint(img.size)
# pprint(type(lbl))
# pprint(lbl)

root = "data/COCO/val2017/"
ann_file = "data/COCO/annotations/instances_val2017.json"
coco = COCODataset(root, ann_file)
for i in np.random.randint(0, len(coco), 25):
  img, (bboxes, cats) = coco[i]
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  img = draw_bboxes(img, bboxes, cats)
  cv2.imshow("Boxes", img)
  cv2.waitKey(0)
cv2.destroyAllWindows()