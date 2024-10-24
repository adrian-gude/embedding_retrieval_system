# Embedding Retrieval System

This project is an embedding-based retrieval system using FAISS and sentence-transformers for efficient semantic search. The system indexes news articles and allows querying for semantically similar content.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [License](#license)

## Prerequisites
- Python 3.7+
- FAISS
- pandas
- sentence-transformers

## Installation
To get started with the project, clone the repository and install the dependencies.

```bash
git clone <repository-url>
cd embedding_retrieval_system
pip install -r requirements.txt
```

You can install the required dependencies using `requirements.txt`:

```
faiss-cpu
pandas
sentence-transformers
```

## Usage

1. **Prepare the Dataset:**

   - Place your news dataset in `data/synthetic_news.csv`.
   - The dataset should contain columns for titles and content.

2. **Create the Index:**

   The FAISS index can be created from the dataset. Run the script to generate embeddings and create the FAISS index:

   ```bash
   python3 retrieval_system/index/create_index.py
   ```

3. **Query the System:**

   After the index is created, you can run queries to retrieve the top N most similar documents based on the query embeddings.

   ```bash
   python3 retrieval_system/search/queries.py
   ```

## File Structure
```
embedding_retrieval_system/
│
├── data/
│   └── synthetic_news.csv            # News dataset
├── retrieval_system/
│   ├── index/
│   │   └── create_index.py           # Script to create FAISS index
│   └── search/
│       └── queries.py                # Script to run queries and retrieve similar news
└── requirements.txt                  # Python dependencies
```

## License
This project is licensed under the MIT License.
