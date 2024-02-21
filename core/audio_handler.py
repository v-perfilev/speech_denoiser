import os

import torch
import torchaudio
from torchaudio.transforms import Resample, Spectrogram


class AudioHandler:
    n_fft = 430
    hop_length = 160

    def __init__(self, rate=16000, chunk_size=14000):
        self.rate = rate
        self.chunk_size = chunk_size

    def load_audio(self, file_path):
        sample, sample_rate = torchaudio.load(file_path, normalize=True)
        return self.prepare_audio(sample, sample_rate)

    def prepare_audio(self, sample, sample_rate):
        if sample.dim() > 1 and sample.shape[0] == 2:
            sample = sample.mean(dim=0, keepdim=True)
        if sample.dim() == 1:
            sample = sample.unsqueeze(0)
        #
        if sample_rate != self.rate:
            resample_transform = Resample(orig_freq=sample_rate, new_freq=self.rate)
            sample = resample_transform(sample)
            sample_rate = self.rate

        return sample, sample_rate

    def mix_audio_samples(self, main_waveform, background_waveform, background_volume):
        background_waveform *= background_volume

        if main_waveform.shape[1] > background_waveform.shape[1]:
            repeat_times = main_waveform.shape[1] // background_waveform.shape[1] + 1
            background_waveform = background_waveform.repeat(1, repeat_times)
        background_waveform = background_waveform[:, :main_waveform.shape[1]]

        mixed_waveform = main_waveform + background_waveform

        return mixed_waveform

    def divide_audio(self, audio):
        chunks = audio.unfold(0, self.chunk_size, self.chunk_size).contiguous()

        processed_chunks = []
        for chunk in chunks:
            if chunk.size(0) < self.chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, self.chunk_size - chunk.size(0)))
            processed_chunks.append(chunk)
        return processed_chunks

    def compile_audio(self, chunks):
        return torch.cat(chunks, dim=0)

    def sample_to_spectrogram(self, sample):
        spectrogram = Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length)
        return spectrogram(sample)

    def spectrogram_to_sample(self, spectrogram):
        griffin_lim = torchaudio.transforms.GriffinLim(n_fft=self.n_fft, hop_length=self.hop_length)
        return griffin_lim(spectrogram)

    def save_sample(self, audio_data, filename, path="target"):
        os.makedirs(path, exist_ok=True)
        torchaudio.save(path + "/" + filename, audio_data, self.rate)
