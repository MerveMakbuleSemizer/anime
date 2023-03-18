import streamlit as st 
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Anime Recommender",
    page_icon="https://www.asialogy.com/wp-content/uploads/anime-nedir-nasil-yapilir.jpg"
    )
st.title("Covid Survive Classifier Project")

st.markdown("This website was created to predict whether a patient will succumb to Covid-19 disease based on some current symptoms and medical history. The model was trained by the dataset provided by the Mexican government.")
st.image("https://www.asialogy.com/wp-content/uploads/anime-nedir-nasil-yapilir.jpg")
st.sidebar.markdown("**Choose** User ID")
# anime = pd.read_csv("anime.csv")
# title = st.sidebar.selectbox("Title",(sorted(anime.name,reverse=True)))
User_ID = np.loadtxt('user_IDs.txt', dtype=int)
userID = st.sidebar.selectbox("User ID",(sorted(User_ID)))

import detect_similars
recommendations = detect_similars.detect_similar_users(userID)

count = 1
for anime in recommendations.index:
    st.markdown(str(count)+ ". " + anime)
    count += 1
