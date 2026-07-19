import streamlit as st
import numpy as np
import pandas as pd
import pickle
import requests
import io
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ---------------- CONFIGURATION ----------------
BASE_URL = "https://huggingface.co/datasets/ShaliniSaurav/book-data-files/resolve/main/"
HEADERS = {"User-Agent": "Mozilla/5.0"}

st.title("📚 Book Recommendation System")

# ---------------- DATA LOADING (Cached) ----------------
@st.cache_data
def load_data():
    def get_df_from_url(url):
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            return pd.read_csv(io.StringIO(response.text), low_memory=False)
        return None
    
    books = get_df_from_url(BASE_URL + "Books.csv")
    ratings = get_df_from_url(BASE_URL + "Ratings.csv")
    return books, ratings

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_embeddings():
    response = requests.get(BASE_URL + "embeddings.pkl", headers=HEADERS)
    return pickle.load(io.BytesIO(response.content))

# ---------------- MODEL BUILDING (Cached) ----------------
@st.cache_data
def build_popularity_model(books, ratings):
    ratings_with_name = ratings.merge(books, on="ISBN")
    
    # Calculate stats
    num_rating_df = ratings_with_name.groupby("Book-Title")["Book-Rating"].count().reset_index()
    num_rating_df.rename(columns={"Book-Rating": "num_ratings"}, inplace=True)
    
    avg_rating_df = ratings_with_name.groupby("Book-Title")["Book-Rating"].mean().reset_index()
    avg_rating_df.rename(columns={"Book-Rating": "avg_rating"}, inplace=True)
    
    # Merge and filter
    popular_df = num_rating_df.merge(avg_rating_df, on="Book-Title")
    popular_df = popular_df[popular_df["num_ratings"] >= 50].sort_values("avg_rating", ascending=False)
    
    return popular_df.merge(books, on="Book-Title").drop_duplicates("Book-Title").head(20)

# ---------------- MAIN EXECUTION ----------------
books, ratings = load_data()

if books is not None and ratings is not None:
    model = load_model()
    embeddings, unique_books = load_embeddings()
    popular_df = build_popularity_model(books, ratings)

    option = st.selectbox("Choose Option", ["Popular Books", "Recommend"])
    
    if option == "Popular Books":
        st.subheader("Top 20 Rated Books")
        for _, row in popular_df.iterrows():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(row["Image-URL-M"], width=80)
            with col2:
                st.write(f"**{row['Book-Title']}**")
                st.write(f"Author: {row['Book-Author']} | ⭐ {round(row['avg_rating'], 2)} ({row['num_ratings']} votes)")
    
    else:
        book_name = st.text_input("Enter Book Name")
        if st.button("Recommend"):
            st.write("Searching...")
            query_vec = model.encode([book_name])
            sims = cosine_similarity(query_vec, embeddings)[0]
            idx = np.argsort(sims)[::-1][1:6]
            
            for i in idx:
                row = unique_books.iloc[i]
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.image(row["Image-URL-M"], width=80)
                with col2:
                    st.write(f"**{row['Book-Title']}**")
                    st.write(f"Author: {row['Book-Author']}")