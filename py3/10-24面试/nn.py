import torch.nn as nn

def __init__(self):
    super().__init__()

    self.conv1=nn.Conv2d(128,64,kernel_size=1,stride=1)
    self.bn1=nn.BatchNorm2d(64)

    self.conv2=nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
    self.bn2=nn.BatchNorm2d(64)

    self.conv3=nn.Conv2d(64,64,kernel_size=1,stride=1)
