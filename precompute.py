from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle

print("Loading data...")

books = pd.read_csv("books.csv")
unique_books = books.drop_duplicates("Book-Title")

texts = (
    unique_books["Book-Title"].fillna("") +
    " by " +
    unique_books["Book-Author"].fillna("")
).tolist()

print("Loading model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating embeddings (1st time slow hoga)...")

embeddings = model.encode(texts)

print("Saving file...")

pickle.dump((embeddings, unique_books), open("embeddings.pkl", "wb"))

print("Done ✅ embeddings.pkl created")