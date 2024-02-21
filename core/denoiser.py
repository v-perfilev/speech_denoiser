class Denoiser:
    def __init__(self, model, audio_handler):
        self.model = model
        self.audio_handler = audio_handler

    def denoise_sample(self, sample):
        denoised_chunks = []
        chunks = self.audio_handler.divide_audio(sample.squeeze(0))
        for chunk in chunks:
            chunk_spectrogram = self.audio_handler.sample_to_spectrogram(chunk.unsqueeze(0))
            denoised_chunk_spectrogram = self.model(chunk_spectrogram.unsqueeze(0)).squeeze(0)
            denoised_chunk = self.audio_handler.spectrogram_to_sample(denoised_chunk_spectrogram)
            denoised_chunks.append(denoised_chunk)
        return self.audio_handler.compile_audio(denoised_chunks).reshape(1, -1), self.audio_handler.rate
