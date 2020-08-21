import os
from pydub import AudioSegment
from tqdm import tqdm
import pandas as pd
from random import randint


# df = pd.read_csv('Passaros.csv', sep=";")
# df = df.rename(columns = {'arquivo':'fname', 'passaro':'label'})
# df = df[['fname', 'label']]
# df['fname'] = df['fname'].apply(lambda x: x[:-3] + 'wav')
# print(df.head())
# df.to_csv('passaros2.csv')


audio_dir = 'nova/'
for fn in tqdm(os.listdir(audio_dir)):
    if fn[-3:] == 'mp3':
        sound = AudioSegment.from_mp3(audio_dir+fn)
        sound.export(audio_dir+fn[:-3]+'wav', format="wav")
        os.remove(audio_dir+fn)                                                       