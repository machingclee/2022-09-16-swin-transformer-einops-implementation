import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary
from einops.layers.torch import Rearrange, Reduce

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.model = models.swin_t(weights="DEFAULT")
        self.features = self.model.features
        self.norm = self.model.norm
        self.freeze_params()

    def freeze_params(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = x.permute([0, 3, 1, 2])
        return x

if __name__ == "__main__":
    backbone = Backbone()
    t = torch.randn((3, 3, 224, 224))
    out = backbone(t)
    print(out.shape)
