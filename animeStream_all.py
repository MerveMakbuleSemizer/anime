import streamlit as st 
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Anime Recommender",
    page_icon="https://www.asialogy.com/wp-content/uploads/anime-nedir-nasil-yapilir.jpg"
    )
st.title("Anime Recommendar Project")

st.image("https://www.asialogy.com/wp-content/uploads/anime-nedir-nasil-yapilir.jpg")
st.sidebar.markdown("**Choose** User ID")
# anime = pd.read_csv("anime.csv")
# title = st.sidebar.selectbox("Title",(sorted(anime.name,reverse=True)))
User_ID = np.loadtxt('all_users.txt', dtype=int)
userID = st.sidebar.selectbox("User ID",(sorted(User_ID)))

user_based = np.loadtxt('user_IDs.txt',dtype=int)

if userID in user_based:
    import detect_similars
    recommendations = detect_similars.detect_similar_users(userID).index
else:
    import similar_content
    st.sidebar.markdown("**Pleaase select an anime:**")
    anime_name = st.sidebar.selectbox("Title",similar_content.indices.index)
    recommendations = similar_content.get_recommendations(anime_name).values

count = 1
for anime in recommendations:
    st.markdown(str(count)+ ". " + anime)
    count += 1