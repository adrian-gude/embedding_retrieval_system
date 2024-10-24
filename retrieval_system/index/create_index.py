import faiss
import numpy as np


def create_index(vectors, index_path):
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    # quantizer = faiss.IndexFlatL2(vectors.shape[1])
    # index = faiss.IndexIVFFlat(quantizer, vectors.shape[1], nlist=100, m=8, bits=8)
    # index.train(vectors[:50])
    # index.add(vectors)
    print(index.ntotal)
    print(vectors.shape[1])
    faiss.write_index(index, index_path)


def main():
    vectors = np.load("embeddings/synthetic_news_embeddings.npy")
    create_index(vectors, "retrieval_system/index/synthetic_news_index.faiss")


if __name__ == "__main__":
    main()
