# Supreme Court Legal Document Search

An advanced semantic search engine for Indian Supreme Court legal documents.

## Key Features
- **Async PDF parsing**: Rapidly extract text from large legal PDFs.
- **SpaCy chunking**: Intelligent text chunking preserving sentence boundaries.
- **FAISS Vector Index**: Fast similarity search using HNSW indexing.
- **FastAPI**: High-performance REST API wrapper for the search functionality.

## Project Structure
```text
.
├── main.py                 # Minimal FastAPI Inference Server
├── search_engine.py        # Core semantic search, embedding, and FAISS logic
├── fetch_and_preprocess.py # PDF processing and NLP chunking pipelines
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Setup & Installation

1. Clone the repository and navigate to the directory.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Prepare the FAISS index by running the preprocessing and search engine scripts:
   ```bash
   python fetch_and_preprocess.py
   python search_engine.py
   ```
5. Run the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

## Future Roadmap (v2.0)
- **Cross-Encoder re-ranking**: Implement a cross-encoder model to re-rank the top FAISS results for improved precision.
- **Docker Containerization**: Containerize the FastAPI server and indexing pipeline for easier deployment.
