import random

from torch.utils.data import Dataset

from core.audio_handler import AudioHandler


class AudioDataset(Dataset):
    n_fft = 430
    hop_length = 160

    def __init__(self, speech_files, sound_files, speach_format='mp3', sound_format='wav'):
        self.audio_handler = AudioHandler()
        self.noisy_samples, self.clean_samples = self.__create_samples(speech_files,
                                                                       sound_files,
                                                                       speach_format,
                                                                       sound_format)

    def __len__(self):
        return len(self.noisy_samples)

    def __getitem__(self, idx):
        noisy_sample = self.noisy_samples[idx]
        clean_sample = self.clean_samples[idx]
        noisy_spectrogram = self.audio_handler.sample_to_spectrogram(noisy_sample)
        clean_spectrogram = self.audio_handler.sample_to_spectrogram(clean_sample)
        return noisy_spectrogram, clean_spectrogram

    def __create_samples(self, speech_files, sound_files, speech_format, sound_format):
        noisy_samples = []
        clean_samples = []

        for speach_file in speech_files:
            background_volume = random.choice([i / 10 for i in range(1, 5)])
            sound_file = random.choice(sound_files)
            clean_sample, _ = self.audio_handler.load_audio(speach_file, speech_format)
            sound_sample, _ = self.audio_handler.load_audio(sound_file, sound_format)
            noisy_sample = self.audio_handler.mix_audio_samples(clean_sample, sound_sample, background_volume)

            noisy_sample_chunks = self.audio_handler.divide_audio(noisy_sample.squeeze(0))
            for noisy_sample_chunk in noisy_sample_chunks:
                noisy_samples.append(noisy_sample_chunk.unsqueeze(0))

            clean_sample_chunks = self.audio_handler.divide_audio(clean_sample.squeeze(0))
            for clean_sample_chunk in clean_sample_chunks:
                clean_samples.append(clean_sample_chunk.unsqueeze(0))

        return noisy_samples, clean_samples
