import os

import torch
import torch.nn.functional as F
import torchaudio

import config


def load_waveform(file_path, normalize=True):
    """Load a waveform file into a tensor using torchaudio."""
    waveform, rate = torchaudio.load(file_path)
    waveform, rate = prepare_waveform(waveform, rate)
    if normalize:
        waveform = normalize_waveform(waveform)
    return waveform, rate


def prepare_waveform(waveform, rate):
    """Prepare waveform for processing: handle mono/stereo and resample if necessary."""
    # Convert stereo to mono by averaging the two channels
    if waveform.dim() > 1 and waveform.shape[0] == 2:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Ensure the waveform has a batch dimension
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    # Resample to the desired waveform rate if necessary
    if rate != config.SAMPLE_RATE:
        resample_transform = torchaudio.transforms.Resample(orig_freq=rate, new_freq=config.SAMPLE_RATE)
        waveform = resample_transform(waveform)
        rate = config.SAMPLE_RATE
    return waveform, rate


def normalize_waveform(waveform):
    """Normalize the waveform to a range of [-1, 1] based on its maximum absolute value."""
    return waveform / torch.max(torch.abs(waveform))


def mix_waveforms(main_waveform, background_waveform, background_volume):
    """Mix two waveforms with a specified volume for the background."""
    # Apply volume adjustment to the background waveform
    background_waveform *= background_volume
    # Repeat background to match the length of the main waveform
    if main_waveform.shape[1] > background_waveform.shape[1]:
        repeat_times = main_waveform.shape[1] // background_waveform.shape[1] + 1
        background_waveform = background_waveform.repeat(1, repeat_times)
    background_waveform = background_waveform[:, :main_waveform.shape[1]]
    # Mix both waveforms
    mixed_waveform = main_waveform + background_waveform
    return mixed_waveform


def divide_waveform(waveform):
    """Divide waveform into chunks of a predetermined size, padding if necessary."""
    chunks = waveform.squeeze(0).unfold(0, config.CHUNK_SIZE, config.CHUNK_SIZE).contiguous()
    processed_chunks = []
    for chunk in chunks:
        # Pad the chunk if it's less than the full chunk size
        if chunk.size(0) < config.CHUNK_SIZE:
            chunk = torch.nn.functional.pad(chunk, (0, config.CHUNK_SIZE - chunk.size(0)))
        processed_chunks.append(chunk.unsqueeze(0))
    return torch.stack(processed_chunks, dim=0)


def compile_waveform(chunks):
    """Compile chunks of waveform back into a single waveform tensor."""
    return chunks.view(1, -1)


def divide_spectrogram(spectrogram):
    """Divide a spectrogram into fixed-size batches for model processing."""
    spectrogram = spectrogram.squeeze(0)
    total_length = spectrogram.shape[1]
    num_chunks = (total_length + config.TIME_STEPS - 1) // config.TIME_STEPS

    chunks = []
    for i in range(num_chunks):
        start = i * config.TIME_STEPS
        end = start + config.TIME_STEPS
        chunk = spectrogram[:, start:end]
        if chunk.shape[1] < config.TIME_STEPS:
            padding_size = config.TIME_STEPS - chunk.shape[1]
            chunk = F.pad(chunk, (0, padding_size))
        chunk = chunk.unsqueeze(0).unsqueeze(0)
        chunks.append(chunk)

    chunks = torch.cat(chunks, dim=0)
    return chunks, total_length


def compile_spectrogram(chunks, source_length):
    """Reassemble chunks back into the original tensor format."""
    reassembled = torch.cat([chunk for chunk in chunks], dim=2)
    reassembled = reassembled[:, :, :source_length]
    return reassembled


def waveform_to_spectrogram(waveform, transform=None, reshape=True):
    """Convert a waveform to a spectrogram with given FFT, hop length, and window length."""
    spectrogram_fn = torchaudio.transforms.Spectrogram(n_fft=config.N_FFT,
                                                       hop_length=config.HOP_LENGTH,
                                                       win_length=config.WIN_LENGTH,
                                                       power=1.0)
    if transform is not None:
        spectrogram_fn = transform(spectrogram_fn)
    spectrogram = spectrogram_fn(waveform)
    if reshape:
        frequency_bins = config.FREQUENCY_BINS
        spectrogram = spectrogram[:, :frequency_bins, :] if spectrogram.dim() == 3 \
            else spectrogram[:, :, :frequency_bins, :]
    return spectrogram


def spectrogram_to_waveform(spectrogram, transform=None, reshape=True):
    """Convert a spectrogram to a waveform with given FFT, hop length, and window length."""
    inverse_spectrogram_fn = torchaudio.transforms.GriffinLim(n_fft=config.N_FFT,
                                                              hop_length=config.HOP_LENGTH,
                                                              win_length=config.WIN_LENGTH,
                                                              power=1.0,
                                                              n_iter=32)
    if transform is not None:
        inverse_spectrogram_fn = transform(inverse_spectrogram_fn)
    if reshape:
        frequency_bins_pad = int(config.N_FFT / 2 + 1) - config.FREQUENCY_BINS
        spectrogram = F.pad(spectrogram, (0, 0, 0, frequency_bins_pad), 'constant', 0)
    waveform = inverse_spectrogram_fn(spectrogram)
    return waveform


def spectrogram_to_db(spectrogram):
    """Convert a spectrogram to DB with given FFT, hop length, and window length."""
    amplitude_to_db_fn = torchaudio.transforms.AmplitudeToDB()
    return amplitude_to_db_fn(spectrogram.squeeze())


def save_waveform(waveform, filename, path="target"):
    """Save waveform data to a file in a specified directory."""
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, filename)
    torchaudio.save(filepath, waveform, config.SAMPLE_RATE)
