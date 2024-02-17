import random

from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram

from core.audio_handler import AudioHandler


class AudioDataset(Dataset):
    def __init__(self, speech_files, sound_files, speach_format='mp3', sound_format='wav'):
        self.noisy_samples, self.clean_samples = self.__create_samples(speech_files,
                                                                       sound_files,
                                                                       speach_format,
                                                                       sound_format)

    def __len__(self):
        return len(self.noisy_samples)

    def __getitem__(self, idx):
        noisy_sample = self.noisy_samples[idx]
        clean_sample = self.clean_samples[idx]
        noisy_spectrogram = Spectrogram()(noisy_sample)
        clean_spectrogram = Spectrogram()(clean_sample)
        return noisy_spectrogram, clean_spectrogram

    def __create_samples(self, speech_files, sound_files, speech_format, sound_format):
        audio_handler = AudioHandler()

        noisy_samples = []
        clean_samples = []

        for speach_file in speech_files:
            background_volume = random.choice([i / 10 for i in range(1, 11)])
            sound_file = random.choice(sound_files)
            clean_sample, _ = audio_handler.load_audio(speach_file, speech_format)
            sound_sample, _ = audio_handler.load_audio(sound_file, sound_format)
            noisy_sample = audio_handler.mix_audio_samples(clean_sample, sound_sample, background_volume)
            noisy_samples.append(noisy_sample)
            clean_samples.append(clean_sample)

        return noisy_samples, clean_samples
