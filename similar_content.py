import pandas as pd
import numpy as np
import scipy.stats

import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

cols = list(pd.read_csv("anime_new.csv",nrows=1))
anime = pd.read_csv("anime_new.csv",dtype={"anime_id": "int32"},usecols=[i for i in cols if ((i !="members") & (i!="Unnamed: 0"))])
ratings = pd.read_csv("rating.csv")

import re
import string

punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower()) 
anime['genre'] = anime.genre.astype(str).map(punc_lower)

from nltk.tokenize import MWETokenizer 
from nltk.tokenize import word_tokenize
mwe_tokenizer = MWETokenizer([("martial","arts"),("sci","fi"),
                              ("shoujo","ai"),("shounen","ai"),
                              ("slice","of","life"), ("super","power"),
                              ("avant","garde"),("award","winning")])
mwe_tokens = lambda x: mwe_tokenizer.tokenize(word_tokenize(x))
anime['genre'] = anime.genre.map(mwe_tokens)

genre_update = []
for g in anime.genre:
    gen = ""
    for i in g:
        gen += " " + i
    genre_update.append(gen.strip())
anime.genre = genre_update

anime.dropna(subset=["rating"],inplace=True)

anime.reset_index(inplace=True,drop="first")
df_cb = anime.copy() #dataset for content based recommendation

from sklearn.feature_extraction.text import CountVectorizer

cv1 = CountVectorizer()
# Train ve Test veri setlerine CountVectorizer uygulama
cv = cv1.fit_transform(df_cb.genre)

cv_df = pd.DataFrame(cv.toarray(), columns=cv1.get_feature_names_out())

cv_df["anime_id"] = df_cb.anime_id

df1 = pd.merge(cv_df,df_cb[["rating","anime_id"]],on="anime_id")

df1['rating'] = (df1['rating'] - df1['rating'].min()) / (df1['rating'].max() - df1['rating'].min())

cosine_sim = cosine_similarity(df1.drop(["anime_id"],axis=1))

indices = pd.Series(df_cb.index, index=df_cb['name'])

def get_recommendations(anime_name, cosine_sim=cosine_sim):
    
    idx = indices[anime_name] 
    sim_scores = list(enumerate(cosine_sim[idx])) 
    sim_scores.pop(idx)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11] 
    anime_indices = [i[0] for i in sim_scores] 
    return df_cb['name'].iloc[anime_indices]


