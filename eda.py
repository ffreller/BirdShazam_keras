import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa
import definitions

df = pd.read_csv('treino_f.csv')
df.set_index('fname', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('Passaros/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

fig, ax = plt.subplots()
ax.set_title('Class Distribution' ,y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
plt.axis('equal')
plt.show()
df.reset_index(inplace=True)

signals, fft, fbank, mfccs = {}, {}, {}, {}

#nfft é window length (short time fourier). no caso 44100/40

for c in classes:
    file_name = df[df.label == c].iloc[0,0]
    signal, rate = librosa.load('Passaros/'+file_name, sr=44100)
    mask = definitions.envelope(signal, rate, 0.0005)
    signal = signal[mask]
    signals[c] = signal
    fft[c] = definitions.calc_fft(signal, rate)
    bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
    fbank[c] = bank
    mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
    mfccs[c] = mel
    
definitions.plot_signals(signals)
plt.show()

definitions.plot_fft(fft)
plt.show()

definitions.plot_fbank(fbank)
plt.show()

definitions.plot_mfccs(mfccs)
plt.show()

if len(os.listdir('Passaros')) == 0:
    for f in tqdm(df.fname):
        signal, rate = librosa.load('Passaros/'+f, sr=16000)
        mask = definitions.envelope(signal, rate, 0.0005)
        wavfile.write(filename='Passaros/'+f, rate =rate, data=signal[mask])