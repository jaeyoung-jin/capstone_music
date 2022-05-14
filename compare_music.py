# 필요한 모듈 import
import numpy as np
import librosa
import pandas as pd
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

df_30 = pd.read_csv('Project Dataset.csv', index_col = 'musicname & artist')
labels = df_30[['length']]
df_30 = df_30.drop(columns=['length'])

df_30_scaled = sklearn.preprocessing.scale(df_30) 
df_30 = pd.DataFrame(df_30_scaled, columns = df_30.columns)
df_30.head()

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(df_30)
sim_df = pd.DataFrame(similarity, index=labels.index, columns=labels.index)
sim_df.head()

name_artist = 'Invincible - DEAF KEV' # 이것만 변경!!
series = sim_df[name_artist].sort_values(ascending=False)
series = series.drop(name_artist)
series.head(5).to_frame()