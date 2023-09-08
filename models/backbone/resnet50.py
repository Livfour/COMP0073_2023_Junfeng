import torch.nn as nn

from mmpose.models import ResNet

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.resnet = ResNet(depth=50, out_indices=([2]))

    def forward(self, x):
        return self.resnet(x)[0]