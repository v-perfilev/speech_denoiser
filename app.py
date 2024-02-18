from io import BytesIO

import numpy as np
import pyaudio
import soundfile as sf
import torchaudio
from torchaudio.transforms import Spectrogram

from core.audio_handler import AudioHandler
from core.audio_model import AudioModel

audio_handler = AudioHandler()
model = AudioModel()
model.load()
model.eval()


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


def save_audio(audio_data, audio_rate, filename="target/denoised_output.wav"):
    torchaudio.save(filename, audio_data, audio_rate)
    print(f"Audio saved as {filename}")


class Recorder:
    rate = 44100
    channels = 1
    chunk_size = 1024
    n_fft = 430
    hop_length = 160
    audio_file = None

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
        self.audio_file = audio_file

    def denoise_audio(self):
        samples, _ = audio_handler.load_audio(self.audio_file, 'wav')

        griffin_lim = torchaudio.transforms.GriffinLim(n_fft=self.n_fft, hop_length=self.hop_length)

        denoised_chunks = []
        chunks = audio_handler.divide_audio(samples.squeeze(0))
        for chunk in chunks:
            spectrogram = Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length)
            chunk_spectrogram = spectrogram(chunk.unsqueeze(0))
            denoised_chunk_spectrogram = model(chunk_spectrogram)
            denoised_chunk = griffin_lim(denoised_chunk_spectrogram)
            denoised_chunks.append(denoised_chunk)
        denoised_audio_tensor = audio_handler.compile_audio(denoised_chunks).reshape(1, -1)
        return denoised_audio_tensor, self.rate


if __name__ == "__main__":
    list_microphones()
    selected_microphone_idx = select_microphone()

    recorder = Recorder()
    recorder.record_audio(duration=5, microphone_idx=selected_microphone_idx)
    denoised_audio, rate = recorder.denoise_audio()
    save_audio(denoised_audio, rate)
