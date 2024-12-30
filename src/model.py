
import torch
import torch.nn as nn
import torch.nn.functional as F


# Building the CNN Class:
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()

    # Encoder
    self.conv1 = nn.Conv2d(3, 16, 5) # Output: 16 * 60 * 60
    self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # Output: 32 * 60 * 60
    self.pool = nn.MaxPool2d(2, 2)    # Output: 32 * 30 * 30

    self.conv3 = nn.Conv2d(32, 64, 5) # Output: 64 * 26 * 26
    self.conv4 = nn.Conv2d(64, 128, 3) # Output: 128 * 24 * 24
    self.pool = nn.MaxPool2d(2, 2)    # Output: 128 * 12 * 12

    # Decoder
    self.upSample1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2) # Output: 128 * 24 * 24
    self.transConv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1) # Output: 64 * 26 * 26
    self.transConv2 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=1) # Output: 32 * 30 * 30

    self.upSample2 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2) # Output: 32 * 60 * 60
    self.transConv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1) # Output: 16 * 60 * 60
    self.transConv4 = nn.ConvTranspose2d(16, 3, kernel_size=5, stride=1) # Output: 3 * 64 * 64

  def forward(self, x):
    # Encoding
    out = F.relu(self.conv1(x))
    out = F.relu(self.conv2(out))
    saved_out = torch.clone(out) # 32 * 60 * 60
    out = self.pool(out)

    out = F.relu(self.conv3(out))
    out = F.relu(self.conv4(out))
    out = self.pool(out)

    # Decoding
    out = self.upSample1(out)
    out = self.transConv1(out)
    out = self.transConv2(out)

    out = self.upSample2(out) + saved_out # 32 * 60 * 60
    out = self.transConv3(out)
    out = self.transConv4(out) # 3 * 64 * 64

    return out
