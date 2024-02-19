import os
import random

import torch
from torch.utils.data import Dataset, random_split, DataLoader

from core.audio_handler import AudioHandler


def load_dataset(filename="dataset.pt", path="target"):
    return torch.load(path + "/" + filename)


class AudioDataset(Dataset):
    batch_size = 1
    use_mps = False

    def __init__(self, speech_files=None, sound_files=None):
        self.audio_handler = AudioHandler()
        self.noisy_samples, self.clean_samples = self.__create_samples(speech_files, sound_files)

    def __len__(self):
        return len(self.noisy_samples)

    def __getitem__(self, idx):
        noisy_sample = self.noisy_samples[idx]
        clean_sample = self.clean_samples[idx]
        noisy_spectrogram = self.audio_handler.sample_to_spectrogram(noisy_sample)
        clean_spectrogram = self.audio_handler.sample_to_spectrogram(clean_sample)
        return noisy_spectrogram, clean_spectrogram

    def save(self, filename="dataset.pt", path="target"):
        os.makedirs(path, exist_ok=True)
        torch.save(self, path + "/" + filename)

    def configure(self, batch_size=32, use_mps=False):
        self.batch_size = batch_size
        self.use_mps = use_mps
        return self

    def split_into_data_loaders(self, count=None, train_ratio=0.8):
        dataset = self

        if count is not None and count < len(dataset):
            dataset, _ = random_split(self, [count, len(dataset) - count])

        dataset_size = len(dataset)
        train_size = round((dataset_size * train_ratio) / self.batch_size) * self.batch_size

        train_dataset, val_dataset = random_split(dataset, [train_size, dataset_size - train_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.__collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.__collate_fn)

        return train_loader, val_loader

    def __create_samples(self, speech_files, sound_files):
        noisy_samples = []
        clean_samples = []

        for speach_file in speech_files:
            background_volume = random.choice([i / 10 + 0.2 for i in range(1, 5)])
            sound_file = random.choice(sound_files)
            clean_sample, _ = self.audio_handler.load_audio(speach_file)
            sound_sample, _ = self.audio_handler.load_audio(sound_file)
            noisy_sample = self.audio_handler.mix_audio_samples(clean_sample, sound_sample, background_volume)

            self.audio_handler.save_audio(clean_sample, "test_clean_sample.wav")
            self.audio_handler.save_audio(noisy_sample, "test_noisy_sample.wav")

            noisy_sample_chunks = self.audio_handler.divide_audio(noisy_sample.squeeze(0))
            for noisy_sample_chunk in noisy_sample_chunks:
                noisy_samples.append(noisy_sample_chunk.unsqueeze(0))

            clean_sample_chunks = self.audio_handler.divide_audio(clean_sample.squeeze(0))
            for clean_sample_chunk in clean_sample_chunks:
                clean_samples.append(clean_sample_chunk.unsqueeze(0))

        return noisy_samples, clean_samples

    def __collate_fn(self, batch):
        device = torch.device("mps" if self.use_mps and torch.backends.mps.is_available() else "cpu")
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch = [x.to(device).to(torch.float32) for x in batch]
        return batch
