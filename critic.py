import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)
def flatten(x):
    return to_var(x.view(x.size(0), -1))
class critic_FC(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(critic_FC, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(image_size, h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, 20),
            nn.LeakyReLU(0.2),
            nn.Linear(20, 1),
        )

    def forward(self, x,att_type):
        #do something if cnn comes in .flatten it
        x=flatten(x)
        h = self.encoder(x)

        return h