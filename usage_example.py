from io import BytesIO

import numpy as np
import pyaudio
import soundfile as sf
import torch

from core.audio_handler import AudioHandler

model_path = "../models/speech_denoiser_model.pth"
audio_handler = AudioHandler()
model = torch.load(model_path)
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


class Recorder:
    rate = 44100
    channels = 1
    chunk_size = 1024
    audio = None

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

        self.audio = audio_handler.load_audio(audio_file)

    def extract_audio(self):
        samples, _ = self.audio
        return samples, self.rate

    def denoise_audio(self):
        samples, _ = self.audio

        denoised_chunks = []
        chunks = audio_handler.divide_audio(samples.squeeze(0))
        for chunk in chunks:
            chunk_spectrogram = audio_handler.sample_to_spectrogram(chunk.unsqueeze(0))
            denoised_chunk_spectrogram = model(chunk_spectrogram.unsqueeze(0)).squeeze(0)
            denoised_chunk = audio_handler.spectrogram_to_sample(denoised_chunk_spectrogram)
            denoised_chunks.append(denoised_chunk)
        denoised_audio_tensor = audio_handler.compile_audio(denoised_chunks).reshape(1, -1)
        return denoised_audio_tensor, self.rate


if __name__ == "__main__":
    list_microphones()
    selected_microphone_idx = select_microphone()

    recorder = Recorder()
    recorder.record_audio(duration=3, microphone_idx=selected_microphone_idx)
    source_audio, _ = recorder.extract_audio()
    denoised_audio, _ = recorder.denoise_audio()
    audio_handler.save_audio(source_audio, filename="target/source_sample.wav")
    audio_handler.save_audio(denoised_audio, filename="target/denoised_sample.wav")
