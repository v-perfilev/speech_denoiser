import torch
import torch.nn as nn


class SISDRLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(SISDRLoss, self).__init__()
        self.eps = eps

    def forward(self, outputs, targets):
        """Calculates the SI-SDR loss specifically designed for magnitude spectrograms."""
        # Ensure the target and estimate have the same shape
        assert outputs.shape == targets.shape, "Target and estimate must have the same shape"

        # Reshape to merge batch and channel dimensions for independent processing of each spectrogram
        outputs_flat = outputs.reshape(-1, outputs.shape[-2], outputs.shape[-1])
        targets_flat = targets.reshape(-1, targets.shape[-2], targets.shape[-1])

        # Compute the dot product and energy per frequency bin
        dot_product = torch.sum(targets_flat * outputs_flat, dim=2, keepdim=True)
        energy_target = torch.sum(targets_flat ** 2, dim=2, keepdim=True)

        # Compute the scaling factor for the target signal
        alpha = dot_product / (energy_target + self.eps)

        # Compute the scaled target
        scaled_target = alpha * targets_flat

        # Compute the error signal
        e_noise = outputs_flat - scaled_target

        # Compute the energy of target and noise per frequency bin
        target_energy = torch.sum(scaled_target ** 2, dim=2)
        noise_energy = torch.sum(e_noise ** 2, dim=2)

        # Compute the SI-SDR for each frequency bin
        si_sdr = 10 * torch.log10((target_energy + self.eps) / (noise_energy + self.eps))

        # Average over all bins and then over all examples
        si_sdr = si_sdr.mean()

        # We return negative SI-SDR
        return -si_sdr
