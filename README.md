# Speech Denoiser

This app utilizes PyTorch to denoise human speech, distinguishing it from background noise.

For training the model, speech datasets from Mozilla Common Voice and environmental sounds from UrbanSound8K were used.

## Quick Start

1. Clone the repository:

```bash
git clone https://github.com/v-perfilev/speech_denoiser.git
```

2. Install the required packages:


```bash
pip install -r requirements.txt
```


3. Copy dataset with clean and noisy sound samples into the `../_datasets/` directory.
   To generate datasets you can use my another project https://github.com/v-perfilev/audio_dataset_handler.git.


4. Train the model by running the `model_training.ipynb` notebook.


5. Run the app:

```bash
python usage_example.py
```

## Features

- Real-time speech detection using a pretrained neural network model.
- Supports multiple microphone inputs.
- Lightweight and easy to deploy.

## Requirements

- ffmpeg (!!!)
- numpy
- matplotlib
- torchaudio
- pyaudio
- soundfile
- torch
