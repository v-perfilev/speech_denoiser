import torch.nn.functional as F
from torch import nn


class AudioModel(nn.Module):
    dropout_rate = 0.5

    def __init__(self, in_channels=1, out_channels=1):
        super(AudioModel, self).__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Decoder
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv3 = nn.ConvTranspose2d(32, out_channels, kernel_size=2, stride=2)

        # Skip connection
        self.skip_conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        # Encoder

        enc1 = self.enc_conv1(x)
        enc1 = F.relu(enc1)
        enc1 = self.dropout(enc1)
        enc1 = self.pool(enc1)

        skip_output = self.skip_conv1(enc1)
        skip_output = F.relu(skip_output)
        skip_output = self.dropout(skip_output)

        enc2 = self.enc_conv2(enc1)
        enc2 = F.relu(enc2)
        enc2 = self.dropout(enc2)
        enc2 = self.pool(enc2)

        enc3 = self.enc_conv3(enc2)
        enc3 = F.relu(enc3)
        enc3 = self.dropout(enc3)
        enc3 = self.pool(enc3)

        # Decoder
        dec1 = self.dec_conv1(enc3)
        dec1 = F.relu(dec1)
        dec1 = self.dropout(dec1)

        dec2 = self.dec_conv2(dec1)
        dec2 = F.relu(dec2)
        dec2 = self.dropout(dec2)

        dec3 = self.dec_conv3(dec2 + skip_output)
        dec3 = F.relu(dec3)
        dec3 = self.dropout(dec3)

        return dec3

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
