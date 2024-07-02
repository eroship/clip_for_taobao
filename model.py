import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet(nn.Module):
    def __init__(self, out_dim, pretrained:bool=False):
        super.__init__()
        self.backbone = resnet18(pretrained=pretrained)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x