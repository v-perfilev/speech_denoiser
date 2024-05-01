import numpy as np
from matplotlib import pyplot as plt

import config
from utils.audio_utils import waveform_to_spectrogram, spectrogram_to_db


def show_spectrogram(value, title):
    if value.dim() == 1:
        value = value.unsqueeze(0)

    stft = waveform_to_spectrogram(value, reshape=False)
    db = spectrogram_to_db(stft)
    db = db.detach().numpy()

    plt.figure(figsize=(10, 4))
    plt.imshow(db, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=-80)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time steps')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()


def show_waveform(value, title):
    if value.dim() > 1:
        value = value.squeeze(0)
    value = value.detach().numpy()

    time = np.linspace(0, len(value) / config.SAMPLE_RATE, num=len(value))

    plt.figure(figsize=(10, 4))
    plt.plot(time, value)
    plt.title(title)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
