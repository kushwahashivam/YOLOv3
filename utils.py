import os
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision as tv

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, ann_file):
        self.root = root
        self.ann_file = ann_file
        self.cocods = tv.datasets.CocoDetection(self.root, self.ann_file)
        