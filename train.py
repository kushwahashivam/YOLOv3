import os
import time
from tqdm import tqdm, trange
import numpy as np
import torch
import torchvision as tv
from torch.utils.tensorboard import SummaryWriter

from models import YOLOv3
from config import img_size, coco_cat2int, coco_int2cat, cat2cat, cat2int, int2cat, anchors, num_classes

data_dir = "data/COCO/cleaned/val2017/"
model_dir = "model/yolo.pt"
optim_dir = "model/optim.pt"
gs_dir = "summary/gs.pkl"

gs = 1
model = YOLOv3()
optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=.001)
