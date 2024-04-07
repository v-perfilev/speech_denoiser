import os

import torch
import torchaudio
from torchaudio.transforms import Resample, Spectrogram


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

    def save_audio(self, audio_data, filename, path="target"):
        os.makedirs(path, exist_ok=True)
        torchaudio.save(path + "/" + filename, audio_data, self.target_sample_rate)
