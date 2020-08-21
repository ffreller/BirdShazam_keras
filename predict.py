from scipy.io import wavfile
import pandas as pd
import numpy as np
from keras.models import load_model
from tqdm import tqdm
from python_speech_features import mfcc
import pickle
import os
from sklearn import metrics

def build_predictions(audio_dir):
    y_true, y_pred = [], []
    fn_prob = {}
    
    print('\nExtracting features from audio')
    for fn in tqdm(os.listdir(audio_dir)):
        rate, wav = wavfile.read(os.path.join(audio_dir, fn))
        label = fn2class[fn]
        c = classes.index(label)
        y_prob = []
        
        for i in range(0, wav.shape[0]-config.step, config.step):
            sample = wav[i:i+config.step]
            X = mfcc(sample, rate, numcep=config.nfeat,
                        nfilt=config.nfilt, nfft=config.nfft)
            X = (X - config.min) / (config.max - config.min)
        
            if config.mode == 'conv':
                X = X.reshape(1, X.shape[0], X.shape[1], 1)
#            elif config.mode == 'time':
#                X = np.expand_dims(X, axis=0)
            y_hat = model.predict(X)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
            y_true.append(c)
        
        fn_prob[fn] = np.mean(y_prob, axis=0).flatten()
        
    return y_true, y_pred, fn_prob
            
df = pd.read_csv('passaros2.csv')
classes = list(np.unique(df.label))
fn2class = dict(zip(df.fname, df.label))
p_path = os.path.join('pickles','conv.p')

with open(p_path, 'rb') as handle:
    config = pickle.load(handle)
    
model = load_model(config.model_path)

y_true, y_pred, fn_prob = build_predictions('clean_passaros')
accuracy = metrics.accuracy_score(y_true, y_pred)
#precision = metrics.precision_score(y_true, y_pred)
#recall = metrics.recall_score(y_true, y_pred)
#f1 = metrics.f1_score(y_true, y_pred)
#auprc = metrics.average_precision_score(y_true, y_pred)

y_probs = []
for i, row in df.iterrows():
    y_prob = fn_prob[row.fname]
    y_probs.append(y_prob)
    for cl, p in zip(classes, y_prob):
        df.at[i, cl] = p

y_pred = [classes[np.argmax(y)] for y in y_probs]
df['y_pred'] = y_pred
print(y_pred)

df.to_csv('predicitons.csv', index=False)