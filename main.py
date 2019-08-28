import os
import time
from tqdm import tqdm, trange
import numpy as np
import cv2
import torch
import torchvision as tv

from models import YOLOv3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = tv.models.resnet34().to(device).eval()
x = torch.randn(1, 3, 640, 480).to(device)
with torch.no_grad():
    for _ in trange(100):
        _ = model(x)

children = []
for i in range(1, len(list(model.children()))-1):
  children.append(torch.nn.Sequential(*list(model.children())[:i]).to(device))

print("Input shape: ", x.shape)
for i in range(len(children)):
  y = children[i](x)
  print("Output shape at ", i+1, " layer: ", y.shape)

print("\nOutput size here: ", torch.nn.Sequential(*list(model.children())[:6]).to(device)(x).shape)

yolo = YOLOv3().to(device).eval()
inp = torch.randn(1, 3, 480, 640).to(device)
pred = yolo(inp)
print(pred.shape)
with torch.no_grad():
  for _ in trange(100):
    pred = yolo(inp)