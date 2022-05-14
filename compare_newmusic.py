# 필요한 모듈 import
import numpy as np
import librosa
import pandas as pd
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
# import IPython.display as ipd

# 사용자가 유사도를 비교하고자 하는 음악파일
filename = "/content/drive/MyDrive/캡스톤/데이터/비교할 음악/Whole (Rob Gasser Remix) - Chime & Adam Tell" 

# 유사도를 비교하고자 하는 음악의 특징 뽑기
y, sr = librosa.load(filename + ".wav")
rms = librosa.feature.rms(y)
bpm, _ = librosa.beat.beat_track(y) 
zero_crossings = librosa.zero_crossings(y)
harm, perc = librosa.effects.hpss(y)
spec = librosa.feature.spectral_centroid(y)
band = librosa.feature.spectral_bandwidth(y)
roll = librosa.feature.spectral_rolloff(y)
chroma = librosa.feature.chroma_stft(y)
mfccs = librosa.feature.mfcc(y)

result = [filename, len(y), np.mean(rms), np.var(rms), bpm, np.mean(zero_crossings), np.var(zero_crossings), np.mean(harm), np.var(harm), np.mean(perc), np.var(perc), np.mean(spec), np.var(spec), np.mean(band), np.var(band), np.mean(roll), np.var(roll), np.mean(chroma), np.var(chroma)]

for mfcc in mfccs:
    result.append(np.mean(mfcc))
    result.append(np.var(mfcc))

# csv로 저장된 데이터 셋 파일 Data Frame으로 load
df_compare = pd.read_csv('/content/drive/MyDrive/캡스톤/데이터/Project Dataset.csv')

# 유사도를 비교할 음악을 Data Frame에 추가
df_compare.loc[len(df_compare)] = result

# Data Frame 전처리
df_compare = df_compare.set_index("musicname & artist")
labels = df_compare[['length']]
df_compare = df_compare.drop(columns=['length'])
df_compare_scaled = sklearn.preprocessing.scale(df_compare) 
df_compare = pd.DataFrame(df_compare_scaled, columns = df_compare.columns)

# cosine 유사도 검사
similarity = cosine_similarity(df_compare)
sim_df = pd.DataFrame(similarity, index=labels.index, columns=labels.index)
sim_df.head()

# 유사도를 비교하고자 하는 음악과 다른 음악과의 cosine 유사도 결과 내림차순 정렬
name_artist = filename
series = sim_df[name_artist].sort_values(ascending=False)
series = series.drop(name_artist)

# 가장 유사도가 높은 5개 음악 파일의 이름과 cosine 유사도 저장
nearest_cosine_value = list(series[0:5])
nearest_music_name = list(series.index[0:5])

# 가장 유사도가 높은 5개 음악 파일의 이름과 cosine 유사도 출력
for i in range(len(nearest_cosine_value)):
  print(str(i + 1) + "번째로 가장 유사한 노래 : " + str(nearest_music_name[i]) + " (cosine_value : " + str(nearest_cosine_value[i]) + ")")