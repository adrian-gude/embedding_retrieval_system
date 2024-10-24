import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


def parse_news_data(path: str):
    df = pd.read_csv(path)
    return df["content"]


def create_embeddings(df: pd.DataFrame):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(df.to_list(), show_progress_bar=True)
    return embeddings


def save_embeddings(embeddings: np.ndarray, path: str):
    np.save(path, embeddings)


def main():
    df = parse_news_data("data/synthetic_news.csv")
    embeddings = create_embeddings(df)
    save_embeddings(embeddings, "embeddings/synthetic_news_embeddings.npy")


if __name__ == "__main__":
    main()
