from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np
import sys


def queries_to_embeddings(queries):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(queries, show_progress_bar=False)


def search_top_k(index_path: str, query_embeddings: np.ndarray, k: int):

    index = faiss.read_index(index_path)

    D, I = index.search(query_embeddings, k)
    print(D, I)
    return D, I


def main():
    df = pd.read_csv("data/synthetic_news.csv", usecols=["title", "content"])

    queries = ["Quantum Computing", "Historic Agreement"]

    query_embeddings = queries_to_embeddings(queries)

    distances, indices = search_top_k(
        "retrieval_system/index/synthetic_news_index.faiss", query_embeddings, 3
    )

    for query_idx, query in enumerate(queries):
        print(f"\nResults for query: '{query}'")
        print(f"Indices of the closest news articles: {indices[query_idx]}")
        print(f"Distances of the closest news articles: {distances[query_idx]}")

        for idx in indices[query_idx]:
            title = df.loc[idx, "title"]
            content = df.loc[idx, "content"]
            print(f"Title: {title}")
            print(f"Content: {content[:200]}...")  # Show only the first 200 characters
            print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()
