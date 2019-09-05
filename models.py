import torch
import torch.nn as nn
import torchvision as tv

from config import num_classes, anchors


class YOLOv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchor_boxes = len(anchors)
        base = list(tv.models.resnet34(pretrained=True).children())[:6]
        self.base = nn.Sequential(*base).eval()  # Output shape: 128x(H/8)x(w/8)
        del base
        for param in self.base.parameters():
            param.requires_grad = False
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(.1)
        )  # Output shape: 256x(H/8)x(w/8)
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(.1)
        )  # Output shape: 128x(H/8)x(w/8)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(.1)
        )  # Output shape: 128x(H/16)x(w/16)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(.1)
        )  # Output shape: 64x(H/16)x(w/16)
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(.1)
        )  # Output shape: 64x(H/32)x(w/32)
        self.conv6 = nn.Conv2d(64, (self.num_classes+5)*self.num_anchor_boxes, kernel_size=1, stride=1, padding=0)
        # Output shape: ((num_classes+5)*num_anchor_boxes)x(H/32)x(w/32)

    def forward(self, x):
        x = self.base(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        for i in range(self.num_anchor_boxes):
            index_mask = i*(self.num_classes+5)
            # Sigmoid all the outputs except pw, ph
            x[:, index_mask: index_mask+self.num_classes+3] = (
                torch.sigmoid(x[:, index_mask: index_mask+self.num_classes+3])
            )
            # Exponentiate pw, ph
            x[:, index_mask+self.num_classes+3: index_mask+self.num_classes+5] = (
                torch.exp(x[:, index_mask+self.num_classes+3: index_mask+self.num_classes+5])
            )
        return x
