from io import BytesIO

import numpy as np
import pyaudio
import soundfile as sf

from core.audio_handler import AudioHandler
from core.audio_model import AudioModel
from core.denoiser import Denoiser

model = AudioModel()
model.load()
model.eval()

audio_handler = AudioHandler()

denoiser = Denoiser(model, audio_handler)


def list_microphones():
    p = pyaudio.PyAudio()
    print("List of available microphones:")
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev['maxInputChannels'] > 0:
            print(f"{i}: {dev['name']}")
    p.terminate()


def select_microphone():
    while True:
        idx = input("Enter the microphone index: ")
        if idx.isdigit() and 0 <= int(idx) < pyaudio.PyAudio().get_device_count():
            return int(idx)
        else:
            print("Invalid index. Please try again.")


class Recorder:
    rate = 44100
    channels = 1
    chunk_size = 1024
    sample = None

    def record_audio(self, duration, microphone_idx):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=self.channels, rate=self.rate, input=True,
                        frames_per_buffer=self.chunk_size, input_device_index=microphone_idx)

        print("Recording...")
        frames = []
        for _ in range(0, int(self.rate / self.chunk_size * duration)):
            data = stream.read(self.chunk_size)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_file = BytesIO()
        sf.write(audio_file, audio_data, self.rate, format='wav')
        audio_file.seek(0)

        return audio_handler.load_audio(audio_file)


if __name__ == "__main__":
    list_microphones()
    selected_microphone_idx = select_microphone()

    recorder = Recorder()
    noisy_sample, _ = recorder.record_audio(duration=3, microphone_idx=selected_microphone_idx)
    denoised_sample, _ = denoiser.denoise_sample(noisy_sample)
    audio_handler.save_sample(noisy_sample, filename="source_sample.wav")
    audio_handler.save_sample(denoised_sample, filename="denoised_sample.wav")
