import os

import torch
from torch import nn


class AudioModel(nn.Module):
    dropout_rate = 0.3

    def __init__(self, in_channels=1, out_channels=1):
        super(AudioModel, self).__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Decoder
        self.dec_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv2 = nn.ConvTranspose2d(32, out_channels, kernel_size=2, stride=2)

        # Skip connection
        self.skip_conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        # Encoder

        enc1 = self.enc_conv1(x)
        enc1 = nn.ReLU(inplace=True)(enc1)
        enc1 = self.pool(enc1)

        skip_output = self.skip_conv1(enc1)
        skip_output = nn.ReLU(inplace=True)(skip_output)

        enc2 = self.enc_conv2(enc1)
        enc2 = nn.ReLU(inplace=True)(enc2)
        enc2 = self.pool(enc2)

        # Decoder
        dec1 = self.dec_conv1(enc2)
        dec1 = nn.ReLU(inplace=True)(dec1)

        dec1 = dec1 + skip_output

        dec2 = self.dec_conv2(dec1)
        dec2 = nn.ReLU(inplace=True)(dec2)

        return dec2

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
