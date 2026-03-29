import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

st.title("📚 Book Recommendation System")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    books = pd.read_csv("books.csv")
    ratings = pd.read_csv("ratings.csv")
    users = pd.read_csv("users.csv")
    return books, ratings, users

# ---------------- POPULAR BOOKS ----------------
@st.cache_data
def build_popularity_model(books, ratings):
    ratings_with_name = ratings.merge(books, on="ISBN")

    num_rating_df = ratings_with_name.groupby("Book-Title")["Book-Rating"].count().reset_index()
    num_rating_df.rename(columns={"Book-Rating": "num_ratings"}, inplace=True)

    avg_rating_df = ratings_with_name.groupby("Book-Title")["Book-Rating"].mean().reset_index()
    avg_rating_df.rename(columns={"Book-Rating": "avg_rating"}, inplace=True)

    popular_df = num_rating_df.merge(avg_rating_df, on="Book-Title")
    popular_df = popular_df[popular_df["num_ratings"] >= 50]
    popular_df = popular_df.sort_values("avg_rating", ascending=False).head(20)

    popular_df = popular_df.merge(books, on="Book-Title").drop_duplicates("Book-Title")

    return popular_df

# ---------------- COLLAB FILTER ----------------
@st.cache_data
def build_collab_model(books, ratings):
    ratings_with_name = ratings.merge(books, on="ISBN")

    x = ratings_with_name.groupby("User-ID")["Book-Rating"].count()
    active_users = x[x > 50].index

    filtered = ratings_with_name[ratings_with_name["User-ID"].isin(active_users)]

    y = filtered.groupby("Book-Title")["Book-Rating"].count()
    famous_books = y[y >= 20].index

    final_ratings = filtered[filtered["Book-Title"].isin(famous_books)]

    pt = final_ratings.pivot_table(
        index="Book-Title",
        columns="User-ID",
        values="Book-Rating"
    ).fillna(0)

    similarity_scores = cosine_similarity(pt)

    return pt, similarity_scores

# ---------------- LOAD EMBEDDINGS ----------------
@st.cache_resource
def load_embeddings():
    embeddings, unique_books = pickle.load(open("embeddings.pkl", "rb"))
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model, embeddings, unique_books

# ---------------- COLLAB RECOMMEND ----------------
def recommend(book_name, pt, similarity_scores, books):
    if book_name not in pt.index:
        return None

    index = np.where(pt.index == book_name)[0][0]

    similar_items = sorted(
        list(enumerate(similarity_scores[index])),
        key=lambda x: x[1],
        reverse=True
    )[1:9]

    data = []
    for i in similar_items:
        item = pt.index[i[0]]
        temp_df = books[books["Book-Title"] == item].drop_duplicates("Book-Title")

        data.append({
            "title": temp_df["Book-Title"].values[0],
            "author": temp_df["Book-Author"].values[0],
            "image": temp_df["Image-URL-M"].values[0]
        })

    return data

# ---------------- ZERO SHOT RECOMMEND ----------------
def zero_shot_recommend(query, model, embeddings, unique_books):
    query_vec = model.encode([query])
    sims = cosine_similarity(query_vec, embeddings)[0]

    idx = np.argsort(sims)[::-1][:8]

    results = []
    for i in idx:
        row = unique_books.iloc[i]
        results.append({
            "title": row["Book-Title"],
            "author": row["Book-Author"],
            "image": row["Image-URL-M"]
        })

    return results

# ---------------- MAIN ----------------
books, ratings, users = load_data()

popular_df = build_popularity_model(books, ratings)
pt, sim_scores = build_collab_model(books, ratings)

# 🔥 Load embeddings (important)
model, embeddings, unique_books = load_embeddings()

option = st.selectbox("Choose Option", ["Popular Books", "Recommend"])

# ---------------- POPULAR ----------------
if option == "Popular Books":
    st.subheader("Top Books")

    for _, row in popular_df.iterrows():
        st.image(row["Image-URL-M"], width=100)
        st.write(row["Book-Title"])
        st.write(row["Book-Author"])
        st.write("⭐", round(row["avg_rating"], 2))
        st.write("---")

# ---------------- RECOMMEND ----------------
else:
    book_name = st.text_input("Enter Book Name")

    if st.button("Recommend"):
        result = recommend(book_name, pt, sim_scores, books)

        if result:
            st.success("Showing Similar Books 📖")
            for book in result:
                st.image(book["image"], width=100)
                st.write(book["title"])
                st.write(book["author"])
                st.write("---")

        else:
            st.warning("Book not found → Using AI Recommendation 🤖")

            result = zero_shot_recommend(
                book_name,
                model,
                embeddings,
                unique_books
            )

            for book in result:
                st.image(book["image"], width=100)
                st.write(book["title"])
                st.write(book["author"])
                st.write("---")