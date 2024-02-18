import os

import torch
import torch.nn.functional as F
from torch import nn


class AudioModel(nn.Module):
    dropout_rate = 0.4

    def __init__(self, in_channels=1, out_channels=1, use_mps=False):
        super(AudioModel, self).__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Decoder
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm2d(64)
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.bn5 = nn.BatchNorm2d(32)
        self.dec_conv3 = nn.ConvTranspose2d(32, out_channels, kernel_size=2, stride=2)
        self.bn6 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip_conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.skip_bn1 = nn.BatchNorm2d(32)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)

        if use_mps and torch.backends.mps.is_available():
            device = torch.device("mps")
            self.to(device)
            self.to(torch.float32)

    def forward(self, x):
        # Encoder

        enc1 = self.enc_conv1(x)
        enc1 = self.bn1(enc1)
        enc1 = F.relu(enc1)
        enc1 = self.pool(enc1)

        skip_output = self.skip_conv1(enc1)
        skip_output = self.skip_bn1(skip_output)
        skip_output = F.relu(skip_output)

        enc2 = self.enc_conv2(enc1)
        enc2 = self.bn2(enc2)
        enc2 = F.relu(enc2)
        enc2 = self.pool(enc2)

        enc3 = self.enc_conv3(enc2)
        enc3 = self.bn3(enc3)
        enc3 = F.relu(enc3)
        enc3 = self.pool(enc3)

        # Decoder
        dec1 = self.dec_conv1(enc3)
        dec1 = self.bn4(dec1)
        dec1 = F.relu(dec1)

        dec2 = self.dec_conv2(dec1)
        dec2 = self.bn5(dec2)
        dec2 = F.relu(dec2)

        dec3 = self.dec_conv3(dec2 + skip_output)
        dec3 = self.bn6(dec3)
        dec3 = F.relu(dec3)

        return dec3

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def save(self, filename="audio_model_weights.pth", dict_path="target"):
        os.makedirs(dict_path, exist_ok=True)
        torch.save(self.state_dict(), dict_path + "/" + filename)

    def load(self, filename="audio_model_weights.pth", dict_path="target"):
        self.load_state_dict(torch.load(dict_path + "/" + filename))
        self.eval()
