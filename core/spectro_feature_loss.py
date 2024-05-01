import torch.nn as nn
import torch.nn.functional as F

from utils.audio_utils import spectrogram_to_waveform


class SpectroFeatureLoss(nn.Module):
    def __init__(self, alpha=0.8, transform=None):
        """Initialize the loss module."""
        super(SpectroFeatureLoss, self).__init__()
        self.alpha = alpha
        self.transform = transform
        self.feature_extractor = FeatureExtractor()
        # Optional transformation (e.g., device transfer)
        if transform is not None:
            self.feature_extractor = transform(self.feature_extractor)

    def forward(self, outputs, targets):
        """Calculate the combined loss between predictions and targets."""
        # Ensure the target and estimate have the same shape
        assert outputs.shape == targets.shape, "Target and estimate must have the same shape"

        # L1 loss on the spectrogram
        spectrograms_loss = F.mse_loss(outputs, targets)

        # Convert spectrogram to time domain using inverse STFT
        outputs_waveform = spectrogram_to_waveform(outputs, transform=self.transform)
        targets_waveform = spectrogram_to_waveform(targets, transform=self.transform)
        # Extract waveform features
        outputs_features = self.feature_extractor(outputs_waveform)
        targets_features = self.feature_extractor(targets_waveform)
        # L1 loss on the features
        feature_loss = F.l1_loss(outputs_features, targets_features)

        # # Combine the losses
        combined_loss = self.alpha * spectrograms_loss + (1 - self.alpha) * feature_loss
        return combined_loss


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=65, stride=4, padding=32),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(4),
            nn.Conv1d(16, 32, kernel_size=33, stride=2, padding=16),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(4),
        )

    def forward(self, x):
        x = self.extractor(x)
        return x
