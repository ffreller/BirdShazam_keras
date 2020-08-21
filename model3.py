from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense, BatchNormalization
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
import definitions
import pickle
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os



DROPOUT = 0.5
BATCH_SIZE = 32
EPOCHS = 100
nome = 'Ep100Dr0.4'
#LEARNING_RATE = 0.6

diretorio = 'sem_silencio/'
np.random.seed(42)

def check_data():
    if os.path.isfile(config.p_path):
        print(f'Loading existing data for {config.mode} model')
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None

def build_rand_feat():
    
    tmp = check_data()
    if tmp:
        return tmp.data[0], tmp.data[1], tmp.data[2], tmp.data[3]
    
    X_test, y_test, X_train, y_train = [], [], [], []
    _min, _max = float('inf'), -float('inf')
    print('Treino')
    for _ in tqdm(range(tr_samples)):
        rand_class = np.random.choice(class_dist_tr.index, p=prob_dist_tr)
        file = np.random.choice(df_tr[df_tr.label==rand_class].index)
        rate, wav = wavfile.read(diretorio+file)
        label = df_tr.at[file, 'label']
        
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
        X_train.append(X_sample)
        y_train.append(classes.index(label))
        
    config.min = _min
    config.max = _max
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = (X_train - _min) / (_max - _min)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    y_train = to_categorical(y_train, num_classes=len(classes))

    print('Teste')
    for _ in tqdm(range(te_samples)):
        rand_class = np.random.choice(class_dist_te.index, p=prob_dist_te)
        file = np.random.choice(df_te[df_te.label==rand_class].index)
        rate, wav = wavfile.read(diretorio+file)
        label = df_te.at[file, 'label']
        
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
        X_test.append(X_sample)
        y_test.append(classes.index(label))
    config.min = _min
    config.max = _max
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = (X_test - _min) / (_max - _min)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    y_test = to_categorical(y_test, num_classes=len(classes))
    
    config.data = (X_train, y_train, X_test, y_test)

    with open('pickles/20Especies', 'wb') as handle:
        pickle.dump(config, handle, protocol=2)
        
    return X_train, y_train, X_test, y_test

#if you have a huge inpute place, make yor strides (2,2)
def get_conv_model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation = 'relu', strides=(1, 1),
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation = 'relu', strides=(1, 1),
                     padding='same'))
    model.add(Conv2D(64, (3, 3), activation = 'relu', strides=(1, 1),
                     padding='same'))
#    model.add(Conv2D(128, (3, 3), activation = 'relu', strides=(1, 1),
#                     padding='same'))
#    
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(DROPOUT))
#    model.add(BatchNormalization())
    model.add(Flatten()) # 1 dimension
    

    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(classes), activation='softmax'))
    
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
    return model
    

df_te = pd.read_csv('teste_f.csv')
df_te.set_index('fname', inplace=True)
df_tr = pd.read_csv('treino_f.csv')
df_tr.set_index('fname', inplace=True) 


for f in df_tr.index:
    rate, signal = wavfile.read(diretorio+f)
    df_tr.at[f, 'length'] = signal.shape[0]/rate
    
for ff in df_te.index:
    rate, signal = wavfile.read(diretorio+ff)
    df_te.at[ff, 'length'] = signal.shape[0]/rate
    
tr_samples = 2 * int(df_tr.length.sum()*10)    
te_samples = 2 * int(df_te.length.sum()*10)        


classes = list(np.unique(df_tr.label))

class_dist_tr = df_tr.groupby(['label'])['length'].mean()
class_dist_te = df_te.groupby(['label'])['length'].mean()

prob_dist_tr = class_dist_tr/ class_dist_tr.sum()
prob_dist_te = class_dist_te / class_dist_te.sum()
#choices = np.random.choice(class_dist.index, p=prob_dist)

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist_tr, labels=class_dist_tr.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show()

config = definitions.Config(mode='conv')
if config.mode == 'conv':
    X_train, y_train, X_test, y_test = build_rand_feat()
    y_flat = np.argmax(y_train , axis=1)
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = get_conv_model()

    
class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)

checkpoint = ModelCheckpoint(f'models/{nome}c.model', monitor='val_acc', verbose=1, mode='max',
                             save_best_only=True, save_weights_only=False, period=1)
#earlystopping = EarlyStopping(monitor='val_acc', mode='max', patience=6, min_delta=0.01)

print(f'DROPOUT: {DROPOUT}')
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
          shuffle=True, callbacks=[checkpoint], validation_data = (X_test, y_test))


model.save(f'models/{nome}f.model')