# -*- coding: utf-8 -*-
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
import definitions
import pickle
from keras.callbacks import ModelCheckpoint
import os


DROPOUT = 0.5
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.5

print(DROPOUT, LEARNING_RATE)

diretorio = 'sem_silencio/'

def check_data():
    if os.path.isfile(config.p_path):
        print(f'Loading existing data for {config.mode} model')
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None

def build_rand_feat():
    # tmp = check_data()
    # if tmp:
    #     return tmp.data[0], tmp.data[1]
    X, y = [], []
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.label==rand_class].index)
        rate, wav = wavfile.read(diretorio+file)
        label = df.at[file, 'label']
        
        Step = int(rate/10)
        
        rand_index = np.random.randint(0, wav.shape[0]-Step)
        sample = wav[rand_index:rand_index+Step]
        
        Nfft = 1
        while Nfft < (0.025 * rate):
            Nfft *= 2
        
        X_sample = mfcc(sample, rate, numcep=config.nfeat,
                        nfilt=config.nfilt, nfft=Nfft)
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample)
        y.append(classes.index(label))
    config.min = _min
    config.max = _max
    X, y = np.array(X), np.array(y)
    X = (X - _min) / (_max - _min)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    y = to_categorical(y, num_classes=len(classes))
    config.data = (X, y)
    
    try:
        with open(config.p_path, 'wb') as handle:
            pickle.dump(config, handle, protocol=2)
    except Exception as e:
        print(e)
        
    return X, y

#if you have a huge npute place, make yor strides (2,2)
def get_conv_model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation = 'relu', strides=(1, 1),
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation = 'relu', strides=(1, 1),
                     padding='same'))
    model.add(Conv2D(64, (3, 3), activation = 'relu', strides=(1, 1),
                     padding='same'))
    model.add(Conv2D(128, (3, 3), activation = 'relu', strides=(1, 1),
                     padding='same'))
    
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(DROPOUT)) # chance de 50% de nó ser desativado
    model.add(Flatten()) # 1 dimension
    

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(classes), activation='softmax'))
    
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
    return model
    

df = pd.read_csv('treino_f.csv')
df.set_index('fname', inplace=True)

for f in df.index:
    rate, signal = wavfile.read(diretorio+f)
    df.at[f, 'length'] = signal.shape[0]/rate
classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

#0.1 segundo
n_samples = 2 * int(df.length.sum()*10)
prob_dist = class_dist / class_dist.sum()
#choices = np.random.choice(class_dist.index, p=prob_dist)

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show()

config = definitions.Config(mode='conv')


if config.mode == 'conv':
    X, y = build_rand_feat()
    y_flat = np.argmax(y , axis=1)
    input_shape = (X.shape[1], X.shape[2], 1)
    model = get_conv_model()
    # else:
    #     loadblablba
    
#elif config.mode == 'time':
#    X, y = build_rand_feat()
#    y_flat = np.argmax(y , axis=1)
#    input_shape = (X.shape[1], X.shape[2])
#    model = get_recurrent_model()
    
class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)

checkpoint = ModelCheckpoint('models/sem_silencio.model', monitor='val_acc', verbose=1, mode='max',
                             save_best_only=True, save_weights_only=False, period=1)

model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True,
          validation_split=0.2, callbacks=[checkpoint])


# cd Desktop/DataScience/PROJETO_INTEGRADOR/Audio-Classification-master

# PARAMETROS PRINCIPAIS:
#épocas, lerning rate, batch size, dropout


model.save('models/conv_final.model')