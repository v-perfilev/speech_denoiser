import os

import torch
from torch import nn


class AudioModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(AudioModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
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
