import os

import torch
from torch import nn


class AudioModel(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, use_mps=False):
        super(AudioModel, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=2, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(32, out_channels, kernel_size=2, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.skip_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        if use_mps and torch.backends.mps.is_available():
            device = torch.device("mps")
            self.to(device)
            self.to(torch.float32)
            print("MPS configuration activated")

    def forward(self, x):
        x = self.encoder1(x)
        skip = self.skip_conv(x)
        x = self.encoder2(x)
        x = self.decoder1(x)
        x = self.decoder2(x + skip)
        return x

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
