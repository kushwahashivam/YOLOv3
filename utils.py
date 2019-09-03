import os
import numpy as np
import cv2
import json
from PIL import Image
import torch
import torchvision as tv

from config import coco_cat2int, coco_int2cat, cat2cat, cat2int, int2cat

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, img_size=640):
        self.root = root
        self.img_size = img_size
        self.imgs_list = os.listdir(os.path.join(self.root, "images"))
        self.lbls_dict = json.load(open(os.path.join(self.root, "annotations.json"), "r"))
    
    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, "images", self.imgs_list[index]))
        lbls = self.lbls_dict[self.imgs_list[index]]
        # Get bounding boxes and categories if category exists in our defined categories
        bboxes = []
        categories = []
        for lbl in lbls:
            category_id = lbl["category_id"]
            category = coco_int2cat[category_id]
            # If category exists in our defined categories then add it to categories
            #and append the bounding boxes
            try:
                category = cat2cat[category]
                category_id = cat2int[category]
                categories.append(category_id)
                bboxes.append(lbl["bbox"])
            except KeyError:
                continue
        # Pad the image with zeros to make it sqaure image
        nimg = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
        nimg.paste(img, img.getbbox())
        return np.array(nimg), (np.array(bboxes), np.array(categories))

def bboxes_to_labels(img, bboxes, categories, num_anchors, anchor_ratios, anchors):
    pass

def draw_bboxes(img, bboxes, categories):
    """
    args:
        img: numpy array (HWC)
        bboxes: bounding boxes in format Nx(X, Y, W, H)
        X, Y are top-left coordinates of bounding box
        categories: int categories for corresponding bbox
    """
    for i, bbox in enumerate(bboxes):
        pt1 = int(bbox[0]), int(bbox[1])
        pt2 = int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])
        img = cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
        text = int2cat[categories[i]]
        cv2.putText(img, text, (pt1[0]+5, pt1[1]+15), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), lineType=cv2.LINE_AA) 
    return img

def yolo_loss():
    pass