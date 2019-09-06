import os
import pickle
from tqdm import tqdm, trange
import numpy as np
import torch
import torchvision as tv
from torch.utils.tensorboard import SummaryWriter

from models import YOLOv3
from config import img_size, coco_cat2int, coco_int2cat, cat2cat, cat2int, int2cat, anchors, num_classes
from utils import COCODataset, bboxes_to_labels, yolo_loss, draw_bboxes

data_dir = "data/COCO/cleaned/val2017/"
model_dir = "model/yolo.pt"
optim_dir = "model/optim.pt"
summary_dir = "summary/"
gs_dir = "summary/gs.pkl"
batch_size = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gs = 1
model = YOLOv3().to(device)
optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=.001)

resume_training = False
if resume_training:
    print("Loading states to resume training...")
    if os.path.exists(gs_dir):
        gs = pickle.load(open(gs_dir))
    else:
        raise FileNotFoundError("Global steps check point not found")
    if os.path.exists(model_dir):
        state_dict = torch.load(model_dir, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError("Model not found")
    if os.path.exists(optim_dir):
        state_dict = torch.load(optim_dir, map_location=device)
        optim.load_state_dict(state_dict)
    else:
        raise FileNotFoundError("Optimizer not found")
    del state_dict

print("Loading data...")
img_transform = tv.transforms.Compose([
    tv.transforms.ToTensor(), 
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
ds = COCODataset(data_dir)
ds_size = len(ds) - (len(ds)%batch_size)
writer = SummaryWriter(summary_dir)

num_epoches = 1
for epoch in range(num_epoches):
    model.train()
    data_indices = np.random.permutation(np.arange(ds_size))
    for i in trange(0, ds_size, batch_size):
        imgs = []
        obj_masks = []
        noobj_masks = []
        labels = []
        for b in range(batch_size):
            img, (bboxes, categories) = ds[data_indices[i+b]]
            obj_mask, noobj_mask, label = bboxes_to_labels(bboxes, categories, 32)
            imgs.append(img_transform(img).numpy())
            obj_masks.append(obj_mask)
            noobj_masks.append(noobj_mask)
            labels.append(label)
        imgs = torch.tensor(imgs).to(device)
        obj_masks = torch.tensor(obj_masks).to(device)
        noobj_masks = torch.tensor(noobj_masks).to(device)
        labels = torch.tensor(labels).to(device)
        optim.zero_grad()
        pred = model(imgs)
        loss = yolo_loss(pred, labels, obj_masks, noobj_masks, device)
        loss.backward()
        optim.step()
        writer.add_scalar("loss/training_loss", loss.item(), gs)
        gs += 1
        break
    else:
        print("Epoch %d completed."%(epoch+1))
writer.flush()
writer.close()