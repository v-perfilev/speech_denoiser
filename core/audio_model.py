import torch
from torch import nn


class AudioModel(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super(AudioModel, self).__init__()
        self.encoder = EncLayer(in_channels, 32)
        self.bottleneck = BottleneckLayer(32, 64)
        self.decoder = DecoderLayer(64, 32)
        self.output = OutputLayer(32, out_channels)

    def forward(self, x):
        enc, pool = self.encoder(x)
        bottleneck = self.bottleneck(pool)
        dec = self.decoder(bottleneck, enc)
        out = self.output(dec)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class EncLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.enc_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )
        self.pool_block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        enc = self.enc_block(x)
        pool = self.pool_block(enc)
        return enc, pool


class BottleneckLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        )
        self.dec_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x1, x2):
        x = self.up_block(x1)
        x = torch.cat((x, x2), dim=1)
        x = self.dec_block(x)
        return x


class OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.block(x)
