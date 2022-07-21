
import torch
from torch import nn
from torchvision.models.vgg import vgg16
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        vgg = vgg16(pretrained=True)
        self.loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in self.loss_network.parameters():
            param.requires_grad = False
        self.mse_loss = nn.MSELoss()

    def forward(self, y, target, fake_label=None):
        return self.mse_loss(
            self.loss_network(y),
            self.loss_network(target)
        )
