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


3. Copy a dataset with speech and environment sound samples to the `../datasets/speech` and `../datasets/sounds`
   directories respectively.


4. Generate and save tensor dataset by running the `generate_dataset.ipynb` Jupiter Notebook. In this and the following
   step you can set is_spectrum_model for using a spectrum based model. A spectrogram based model will be used
   otherwise.


5. Train the model by running the `model_training.ipynb` Jupiter Notebook.


6. Run the app:

```bash
python app.py
```

## Features

- Real-time speech detection using a pretrained neural network model.
- Supports multiple microphone inputs.
- Lightweight and easy to deploy.

## Requirements

- ffmpeg (!!!)
- PyTorch
- PyAudio
- Pydub
- Matplotlib
