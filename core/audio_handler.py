import os

import torch
import torchaudio
from pydub import AudioSegment
from torchaudio.transforms import Resample


def save_dataset(dataset):
    os.makedirs('tmp', exist_ok=True)
    torch.save(dataset, 'tmp/dataset.pt')


def load_dataset():
    return torch.load('tmp/dataset.pt')


class AudioHandler:
    def __init__(self, target_sample_rate=44100):
        self.target_sample_rate = target_sample_rate

    def load_audio(self, file_path, audio_format):
        if audio_format == 'mp3':
            audio = AudioSegment.from_mp3(file_path)
            samples = torch.tensor(audio.get_array_of_samples()).float()
            rate = audio.frame_rate
        else:
            samples, rate = torchaudio.load(file_path)

        if samples.dim() > 1 and samples.shape[0] == 2:
            samples = samples.mean(dim=0, keepdim=True)
        if samples.dim() == 1:
            samples = samples.unsqueeze(0)

        if rate != self.target_sample_rate:
            resample_transform = Resample(orig_freq=rate, new_freq=self.target_sample_rate)
            samples = resample_transform(samples)
            rate = self.target_sample_rate

        return samples, rate

    def mix_audio_samples(self, main_waveform, background_waveform, background_volume):
        background_waveform *= background_volume

        if main_waveform.shape[1] > background_waveform.shape[1]:
            repeat_times = main_waveform.shape[1] // background_waveform.shape[1] + 1
            background_waveform = background_waveform.repeat(1, repeat_times)
        background_waveform = background_waveform[:, :main_waveform.shape[1]]

        mixed_waveform = main_waveform + background_waveform

        return mixed_waveform
