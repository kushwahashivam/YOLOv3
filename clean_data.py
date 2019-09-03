import os
import time
import json
from tqdm import tqdm, trange
import numpy as np
import cv2
import torch
import torchvision as tv

root = "data/COCO/train2017/"
ann_file = "data/COCO/annotations_trainval2017/annotations/instances_train2017.json"
cleaned_dir = "data/COCO/cleaned/train2017/"
ds = tv.datasets.CocoDetection(root, ann_file)
out_of_dim = 0
fnf = 0
count = 1
json_dict = {}
for i in trange(len(ds)):
    try:
        img, lbl = ds[i]
        w, h = img.size
        img.save(os.path.join(cleaned_dir, "images", "image"+str(count)+".jpg"))
        json_dict["image"+str(count)+".jpg"] = lbl
        count += 1
        if w>640 or h>640:
            out_of_dim += 1
    except:
        fnf += 1
        continue
json.dump(json_dict, open(os.path.join(cleaned_dir, "annotations.json"), "w"))
print("Images with dimension greater than 640: ", out_of_dim)
print("Files not found: ", fnf)