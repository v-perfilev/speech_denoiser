import os

import torch
import torchaudio
from torchaudio.transforms import Resample, Spectrogram, InverseSpectrogram


def save_dataset(dataset, filename="dataset.pt"):
    os.makedirs('tmp', exist_ok=True)
    torch.save(dataset, 'tmp/' + filename)


def load_dataset(filename="dataset.pt"):
    return torch.load('tmp/' + filename)


class AudioHandler:
    n_fft = 430
    hop_length = 160

    def __init__(self, target_sample_rate=44100, chunk_size=14000):
        self.target_sample_rate = target_sample_rate
        self.chunk_size = chunk_size

    def load_audio(self, file_path):
        samples, rate = torchaudio.load(file_path)
        return self.prepare_audio(samples, rate)

    def prepare_audio(self, samples, rate):
        if samples.dim() > 1 and samples.shape[0] == 2:
            samples = samples.mean(dim=0, keepdim=True)
        if samples.dim() == 1:
            samples = samples.unsqueeze(0)
        #
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

    def spectral_subtraction(self, sample, noise_level=0.3):
        spectrogram_transform = Spectrogram(n_fft=self.n_fft, power=None)
        inverse_spectrogram_transform = InverseSpectrogram(n_fft=self.n_fft)

        spectrogram = spectrogram_transform(sample)

        noise_estimation = noise_level * torch.rand(spectrogram.size())

        noise_reduced_spectrogram = torch.abs(spectrogram) - noise_estimation.abs()
        noise_reduced_spectrogram = torch.clamp(noise_reduced_spectrogram, min=0)

        phase = torch.angle(spectrogram)
        noise_reduced_spectrogram = noise_reduced_spectrogram * torch.exp(1j * phase)

        return inverse_spectrogram_transform(noise_reduced_spectrogram)

    def save_audio(self, audio_data, audio_rate, filename):
        torchaudio.save(filename, audio_data, audio_rate)
