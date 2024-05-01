from io import BytesIO

import numpy as np
import pyaudio
import soundfile as sf
import torch

import config
from utils.audio_utils import divide_waveform, waveform_to_spectrogram, spectrogram_to_waveform, compile_waveform, \
    save_waveform, load_waveform

model_path = "../_models/speech_denoiser_model.pth"
model = torch.load(model_path, map_location='cpu')
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
    audio = None

    def record_audio(self, duration, microphone_idx):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=config.SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=config.CHUNK_SIZE,
                        input_device_index=microphone_idx)

        print("Recording...")
        frames = []
        for _ in range(0, int(config.SAMPLE_RATE / config.CHUNK_SIZE * duration)):
            data = stream.read(config.CHUNK_SIZE)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_file = BytesIO()
        sf.write(audio_file, audio_data, config.SAMPLE_RATE, format='wav')
        audio_file.seek(0)

        self.audio = load_waveform(audio_file)

    def extract_audio(self):
        waveform, _ = self.audio
        return waveform, config.SAMPLE_RATE

    def denoise_audio(self):
        waveform, _ = self.audio

        chunks = divide_waveform(waveform)
        spectrograms = waveform_to_spectrogram(chunks)
        denoised_spectrograms = model(spectrograms)
        denoised_waveforms = spectrogram_to_waveform(denoised_spectrograms)
        denoised_waveform = compile_waveform(denoised_waveforms)
        return denoised_waveform, config.SAMPLE_RATE


if __name__ == "__main__":
    list_microphones()
    selected_microphone_idx = select_microphone()

    recorder = Recorder()
    recorder.record_audio(duration=3, microphone_idx=selected_microphone_idx)
    source_audio, _ = recorder.extract_audio()
    denoised_audio, _ = recorder.denoise_audio()
    save_waveform(source_audio, filename="source_sample.wav")
    save_waveform(denoised_audio, filename="denoised_sample.wav")
