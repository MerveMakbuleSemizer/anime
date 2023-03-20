import pandas as pd
import numpy as np
import scipy.stats

import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

cols = list(pd.read_csv("anime_new.csv",nrows=1))
anime = pd.read_csv("anime_new.csv",dtype={"anime_id": "int32"},usecols=[i for i in cols if ((i !="members") & (i!="Unnamed: 0"))])
ratings = pd.read_csv("rating.csv")

temp_ratings = ratings.copy()
temp_ratings.drop(index=list(ratings[ratings.rating==-1].index),axis=0,inplace=True)

rating_count = temp_ratings.groupby(["user_id"]).agg(number_of_ratings=("rating","count")).reset_index()

df = pd.merge(anime,pd.merge(temp_ratings,rating_count,on="user_id"),on="anime_id")
df_last = df[df.number_of_ratings>300]
matrix_form = df_last.pivot_table(index= "user_id", columns= "name", values= "rating_y")

# Matrisi normalize ediyorum. Bunu yapabilmek için ortalama puanı sıfıra çekerek önyargıları ortadan kaldırdım. Bunu yapmamın 
# sebebi bazı kullanıcılar genel ortalamının altında değerlendirme yapma eğilimdedirler "tough raters". Rating değerlerini
# normalize ederek herkesi aynı seviyeye çekmiş oluyorum.
matrix_norm = matrix_form.subtract(matrix_form.mean(axis=1),axis="rows")
matrix_norm.head()

user_sim_corr = matrix_norm.T.corr()

# hedef kullanıcımızla benzer davranış gösteren ilk  10 kullanıcımızı korelasyon değerlerine bakarak belirledim.
def detect_similar_users(target_user):
    n = 10
    threshold = 0.4 
    similar_users = user_sim_corr[user_sim_corr[target_user] >= threshold].sort_values(by=target_user,ascending=False)[1:] 
    watched_by_user = matrix_norm[matrix_norm.index == target_user].dropna(axis=1,how="all")
    watched_by_similar_user = matrix_norm[matrix_norm.index.isin(similar_users.index)].dropna(axis=1,how="all")
    watched_by_similar_user.drop(watched_by_user.columns,axis=1, inplace=True, errors='ignore') # benzer kullanıcıların izleyip hedef kullanıcının izlemediği animelerin listesi
    
    #weighted average ile hedef kullanıcımızın izlemediği animelere vereceği puanı tahmin ediyorum. 
    similarity_correlation = user_sim_corr[user_sim_corr.index.isin(watched_by_similar_user.index)][target_user] 
    temp = watched_by_similar_user.copy()
    temp = temp.apply(lambda x:x*similarity_correlation.values)
    sim_sum = (temp*0).apply(lambda x:x+similarity_correlation.values)
    sim_sum.fillna(0)
    sim_sum = sim_sum.apply(np.sum, axis = 0)
    temp.fillna(0)
    temp = temp.apply(np.sum, axis=0)
    result = temp/sim_sum
    result = result+matrix_form.mean() # normalize ettiğim değerleri eskisine çeviriyorum.
    return result.sort_values(ascending=False)[:n]
