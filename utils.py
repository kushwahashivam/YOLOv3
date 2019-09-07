import os
import numpy as np
import cv2
import json
from PIL import Image
import torch
import torchvision as tv

from config import img_size, coco_int2cat, cat2cat, cat2int, int2cat, anchors, num_classes


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs_list = os.listdir(os.path.join(self.root, "images"))
        self.lbls_dict = json.load(open(os.path.join(self.root, "annotations.json"), "r"))

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, "images", self.imgs_list[index]))
        # print(img.mode, img.size, img.getbbox())
        lbls = self.lbls_dict[self.imgs_list[index]]
        # Get bounding boxes and categories if category exists in our defined categories
        bboxes = []
        categories = []
        for lbl in lbls:
            category_id = lbl["category_id"]
            category = coco_int2cat[category_id]
            # If category exists in our defined categories then add it to categories
            # and append the bounding boxes
            try:
                category = cat2cat[category]
                category_id = cat2int[category]
                categories.append(category_id)
                bboxes.append(lbl["bbox"])
            except KeyError:
                continue
        # Pad the image with zeros to make it sqaure image
        nimg = Image.new("RGB", (img_size, img_size), (0, 0, 0))
        nimg.paste(img, (0, 0))
        return np.array(nimg, dtype=np.float32), (np.array(bboxes, dtype=np.float32), np.array(categories))


def bboxes_to_labels(bboxes, categories, downsample_scale):
    m_anchors = np.array(anchors, dtype=np.float32)
    num_anchors = len(anchors)
    if len(bboxes) == 0:
        lambda_obj = np.zeros((num_anchors, img_size//downsample_scale, img_size//downsample_scale), dtype=np.float32)
        lambda_noobj = np.ones((num_anchors, img_size//downsample_scale, img_size//downsample_scale), dtype=np.float32)
        label = np.zeros((num_anchors*(num_classes+5), img_size//downsample_scale, img_size//downsample_scale), dtype=np.float32)
        return lambda_obj, lambda_noobj, label
    # Repeat each anchor box to number of bboxes so as to find one to one IOU
    m_anchors = m_anchors.repeat(len(bboxes), axis=0)
    # Reshape anchors to num_anchors X bboxes_length X 2
    m_anchors = m_anchors.reshape(num_anchors, len(bboxes), 2)
    # Get center x, y coordinates of bboxes
    center_xy = np.column_stack((bboxes[:, 0]+bboxes[:, 2]/2, bboxes[:, 1]+bboxes[:, 3]/2))
    # Convert bboxes into x1, y1, x2, y2 format to calculate IOU
    x1y1x2y2 = np.column_stack((bboxes[:, 0], bboxes[:, 1], bboxes[:, 0]+bboxes[:, 2], bboxes[:, 1]+bboxes[:, 3]))
    # Calculate anchor x1, y1, x2, y2. That is, subtract half of width and height of anchor to get x1, y1
    # and width and height to get x2, y2
    # As there are multiple anchors in our model, we need to calculate anchor x1, y1, x2, y2 separately for them
    # and find IOU separately
    anchor_x1y1x2y2 = []
    for i in range(num_anchors):
        a = m_anchors[i]
        anchor_x1y1 = center_xy - a/2
        anchor_x2y2 = anchor_x1y1 + a
        anchor_xy = np.column_stack((anchor_x1y1, anchor_x2y2))
        anchor_x1y1x2y2.append(anchor_xy)
    anchor_x1y1x2y2 = np.array(anchor_x1y1x2y2, dtype=np.float32)
    # Calculate IOU for each anchor box and stack them column wise to calculate argmax
    ious = []
    for i in range(num_anchors):
        a = anchor_x1y1x2y2[i]
        iou = tv.ops.box_iou(torch.from_numpy(a), torch.from_numpy(x1y1x2y2)).diagonal(0).numpy()
        ious.append(iou)
    ious = np.column_stack(ious)
    # Calculate argmax to decide which anchor is given labels
    argmax_ious = np.argmax(ious, axis=1)
    # Downsample by downsample_scale and Convert center_xy to int format for
    # index during assigning of labels in 3D array
    # Now we need
    #   lambda_obj: object mask for each anchor
    #   lambda_noobj: no object mask for each anchor
    #   label: (center xy coordinates for yolo, width and height of anchor boxes)
    # To calculate center xy coordinates for yolo, subtract its integer value from downsampled center_xy
    obj_mask = np.zeros((num_anchors, img_size//downsample_scale, img_size//downsample_scale), dtype=np.float32)
    noobj_mask = np.ones((num_anchors, img_size//downsample_scale, img_size//downsample_scale), dtype=np.float32)
    label = np.zeros((num_anchors*(num_classes+5), img_size//downsample_scale, img_size//downsample_scale), dtype=np.float32)
    # Iterate over each bbox and assign labels and mask
    for i in range(len(bboxes)):
        center = center_xy[i]
        true_anchor = argmax_ious[i]
        index_mask = true_anchor*(num_classes+5)
        # Set mask values
        obj_mask[true_anchor, int(center[1]/downsample_scale), int(center[0]/downsample_scale)] = 1.
        noobj_mask[true_anchor, int(center[1]/downsample_scale), int(center[0]/downsample_scale)] = 0.
        # Objectness score
        label[index_mask+0, int(center[1]/downsample_scale), int(center[0]/downsample_scale)] = 1.
        # Class label
        label[index_mask+1+categories[i], int(center[1]/downsample_scale), int(center[0]/downsample_scale)] = 1.
        # Centre x
        label[index_mask+1+num_classes, int(center[1]/downsample_scale), int(center[0]/downsample_scale)] = center[0]/downsample_scale - int(center[0]/downsample_scale)
        # Centre y
        label[index_mask+1+num_classes+1, int(center[1]/downsample_scale), int(center[0]/downsample_scale)] = center[1]/downsample_scale - int(center[1]/downsample_scale)
        # Width
        label[index_mask+1+num_classes+2, int(center[1]/downsample_scale), int(center[0]/downsample_scale)] = bboxes[i, 2]
        # Height
        label[index_mask+1+num_classes+1, int(center[1]/downsample_scale), int(center[0]/downsample_scale)] = bboxes[i, 3]
    return obj_mask, noobj_mask, label


def yolo_loss(pred, labels, obj_mask, noobj_mask, device):
    lambda_coord = 5.  # Weight for loss in bounding box
    lambda_noobj = .5  # Weight for loss in confidence for regions with no object
    num_anchors = len(anchors)
    batch_size = pred.size(0)
    # mean_loss = mean_loss + (1/n)*(loss - mean_loss)
    mean_loss = torch.tensor(0.).to(device)
    for batch in range(batch_size):
        loss = torch.tensor(0.).to(device)
        for anchor in range(num_anchors):
            index_mask = anchor*(num_classes+5)
            # Confidence loss
            conf_loss = torch.sum(
                obj_mask[batch, anchor] * ((pred[batch, index_mask] - labels[batch, index_mask])**2)
            )
            conf_loss = conf_loss + lambda_noobj * torch.sum(
                noobj_mask[batch, anchor] * ((pred[batch, index_mask] - labels[batch, index_mask])**2)
            )
            loss = loss + conf_loss
            # print("Confidence loss: ", conf_loss.item())
            # Classification loss
            clsf_loss = torch.sum(
                obj_mask[batch, anchor] * ((pred[batch, index_mask+1: index_mask+num_classes+1] - labels[batch, index_mask+1: index_mask+num_classes+1])**2)
            )
            loss = loss + clsf_loss
            # print("Classification loss: ", clsf_loss.item())
            # Localization loss
            # localization loss x, y
            loc_loss = lambda_coord * torch.sum(
                obj_mask[batch, anchor] * ((pred[batch, index_mask+num_classes+1: index_mask+num_classes+3] - labels[batch, index_mask+num_classes+1: index_mask+num_classes+3])**2)
            )
            # localization loss width
            loc_loss = loc_loss + lambda_coord * torch.sum(
                obj_mask[batch, anchor] * ((torch.sqrt(pred[batch, index_mask+num_classes+3]*(anchors[anchor][0]/img_size)) - torch.sqrt(labels[batch, index_mask+num_classes+3]))**2)
            )
            # localization loss height
            loc_loss = loc_loss + lambda_coord * torch.sum(
                obj_mask[batch, anchor] * ((torch.sqrt(pred[batch, index_mask+num_classes+3]*(anchors[anchor][1]/img_size)) - torch.sqrt(labels[batch, index_mask+num_classes+3]))**2)
            )
            loss = loss + loc_loss
            # print("Localization loss: ", loc_loss.item())
        mean_loss = mean_loss + loss
    return mean_loss/batch_size


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
