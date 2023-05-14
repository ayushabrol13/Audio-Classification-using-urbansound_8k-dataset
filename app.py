import json
import streamlit as st
from streamlit_echarts import st_echarts
from millify import millify
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import librosa
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
import soundfile as sf

import warnings
warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="Audio Classification"
)


class AudioClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        # images, labels = images.to(device), labels.to(device)
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        # images, labels = images.to(device), labels.to(device)
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy_score(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}


class UrbanSound8KModel(AudioClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, output_size),
            nn.Sigmoid()
        )

    def forward(self, x_batch):
        return self.network(x_batch)


class UrbanSound8KModel2(AudioClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.ReLU(),

            nn.Linear(512, 64),
            nn.ReLU(),

            nn.Linear(64, output_size),
            nn.Tanh()
        )

    def forward(self, xb):
        return self.network(xb)


input_size = 40
output_size = 10


def extract_mfcc(path):
    audio, sr = librosa.load(path)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)


def main():
    st.title("Audio Classification")
    st.markdown(
        "This is a web app to classify audio files into 10 different categories.")

    # Upload audio file
    st.header("Upload audio file")
    audio_file = st.file_uploader("Upload audio file", type=['wav'])
    if audio_file is None:
        st.write("Please upload an audio file")
    if audio_file is not None:
        st.audio(audio_file, format='audio/wav')

    # use soundfile library to read in the audio fie
    # data, samplerate = sf.read(audio_file)

    classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
               'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

    # Load model using pickle
    model = pickle.load(open('./models/model2.pkl', 'rb'))

    # Send the model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    df = pd.read_csv('./data/UrbanSound8K.csv')

    # audio, sample_rate = librosa.load(audio_file)
    mfccs = extract_mfcc(audio_file)
    mfccs = torch.from_numpy(mfccs)
    mfccs = mfccs.unsqueeze(0)
    mfccs = mfccs.float()
    mfccs = mfccs.to(device)

    with torch.no_grad():
        output = model(mfccs)
        prediction = torch.argmax(output, dim=1)
        st.write("Predicted Class of the Sound: ", classes[prediction])


if __name__ == "__main__":
    main()
