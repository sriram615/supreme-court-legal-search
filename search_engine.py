"""
search_engine.py
================
Highly optimized semantic search engine for the Indian Supreme Court corpus.
Target Architecture: Intel Mac (x86_64 CPU)

Key Optimizations:
  1. NumPy & PyTorch thread constraint: torch.set_num_threads(4) to align matrix math
     natively with physical CPU cores, avoiding AVX core thrashing.
  2. Memory-Safe streaming: reads the large JSONL database in chunked batches
     of 128 rows, preventing massive memory spikes.
  3. FAISS HNSW indexing: uses IndexHNSWFlat with METRIC_INNER_PRODUCT and normalized L2
     vectors to guarantee sub-millisecond retrieval latency on Intel processors.
  4. Fully object-oriented, typed, and production-grade error handling.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 ─ STANDARD LIBRARY IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 ─ THIRD-PARTY IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("[FATAL] numpy not installed. Run: pip install numpy") from exc

try:
    import torch
except ImportError as exc:
    raise SystemExit("[FATAL] torch not installed. Run: pip install torch") from exc

try:
    import faiss
    # Limit FAISS internal OpenMP threads to 1 to prevent deadlock on macOS CPU
    faiss.omp_set_num_threads(1)
except ImportError as exc:
    raise SystemExit("[FATAL] faiss-cpu not installed. Run: pip install faiss-cpu") from exc

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise SystemExit("[FATAL] sentence-transformers not installed. Run: pip install sentence-transformers") from exc

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 ─ LOGGING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)-8s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("search_engine.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("SC_Search")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 ─ THREAD OPTIMIZATION GATES
# ─────────────────────────────────────────────────────────────────────────────
# Force torch to run on a maximum of 4 physical cores to maximize AVX2 efficiency.
# This prevents hyperthreading overhead and core contention on Intel Core i7/i9 processors.
torch.set_num_threads(4)
logger.info("Intel Mac optimization: set torch thread count to 4.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 ─ CENTRAL SEARCH ENGINE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class LegalSearchEngine:
    """
    Central search infrastructure managing Sentence-Transformer embeddings,
    memory-efficient batch processing, and accelerated FAISS HNSW graph indexing.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_dir: str = "./LAWdata_Corpus_2024",
    ) -> None:
        """
        Initialize the search engine with explicit model name and target folders.

        Args:
            model_name: Name of the SentenceTransformer model to load (dimension 384).
            index_dir: The directory where the FAISS index and metadata mappings reside.
        """
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "legal_precedents.index"
        self.metadata_path = self.index_dir / "metadata_map.json"

        # Explicitly enforce CPU device mapping
        self.device = "cpu"
        logger.info("Loading transformer model: %s on device: %s", self.model_name, self.device)
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            # Ensure model runs with CPU thread tuning
            self.model.to(self.device)
            logger.info("Model loaded successfully. Embedding dimension: %d", self.model.get_sentence_embedding_dimension())
        except Exception as exc:
            logger.error("Failed to load transformer model %s: %s", self.model_name, exc)
            raise

        # Internal index and metadata references
        self.index: Optional[faiss.IndexHNSWFlat] = None
        self.metadata_map: List[Dict[str, Any]] = []

    def build_index(
        self,
        corpus_path: str = "./LAWdata_Corpus_2024/aws_court_chunks.jsonl",
        batch_size: int = 128,
        hnsw_m: int = 16,
        limit: Optional[int] = None,
    ) -> None:
        """
        Stream rows from the JSONL database, encode them into dense vectors in
        memory-safe batches, and build an accelerated FAISS HNSW index.

        Args:
            corpus_path: Path to the input JSONL preprocessed chunks file.
            batch_size: Batch sizing to prevent RAM spikes (default 128).
            hnsw_m: Number of connections per node in HNSW graph (default 16).
            limit: If set, restricts indexing to the first N chunks (e.g. for quick validation).
        """
        t_start = time.perf_counter()
        corpus_file = Path(corpus_path)
        if not corpus_file.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_file.resolve()}")

        self.index_dir.mkdir(parents=True, exist_ok=True)
        npy_cache_path = self.index_dir / "embeddings.npy"

        # Check if pre-computed embeddings and metadata map exist
        if limit is None and npy_cache_path.exists() and self.metadata_path.exists():
            logger.info("Found cached embeddings at %s. Fast loading cached state...", npy_cache_path)
            embeddings_matrix = np.load(str(npy_cache_path)).astype("float32")
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata_map = json.load(f)
            total_rows_processed = len(self.metadata_map)
        else:
            logger.info("Building HNSW semantic index from: %s", corpus_file.resolve())
            if limit is not None:
                logger.info("Dry-run build: limit set to first %d rows.", limit)

            # Collect text fragments, parallel chunk IDs, and full metadata mappings
            text_batch: List[str] = []
            meta_batch: List[Dict[str, Any]] = []
            all_embeddings: List[np.ndarray] = []

            total_rows_processed = 0

            # Read JSONL database iteratively in chunks
            with open(corpus_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        text_chunk = row.get("text_chunk", "").strip()
                        if not text_chunk:
                            continue

                        # Maintain a clean copy of metadata mapping
                        metadata = {
                            "chunk_id": row.get("chunk_id", ""),
                            "parent_judgment_id": row.get("parent_judgment_id", ""),
                            "court": row.get("court", "Supreme Court of India"),
                            "tokens_count": row.get("tokens_count", 0),
                            "text_chunk": text_chunk,
                            "extracted_citations": row.get("extracted_citations", []),
                            "language_flag": row.get("language_flag", "en"),
                            "source_pdf": row.get("source_pdf", ""),
                        }

                        text_batch.append(text_chunk)
                        meta_batch.append(metadata)

                        # Trigger batch encoding when size matches limit
                        if len(text_batch) == batch_size:
                            embeddings = self.model.encode(
                                text_batch,
                                batch_size=batch_size,
                                show_progress_bar=False,
                                convert_to_numpy=True,
                                normalize_embeddings=True,  # Normalized L2 vectors for cosine similarity
                            )
                            all_embeddings.append(embeddings)
                            self.metadata_map.extend(meta_batch)
                            total_rows_processed += len(text_batch)
                            logger.info("  Processed %d rows...", total_rows_processed)

                            # Clean batch references
                            text_batch.clear()
                            meta_batch.clear()

                        # Exit if limit reached
                        if limit is not None and total_rows_processed >= limit:
                            break

                    except json.JSONDecodeError as exc:
                        logger.warning("Malformed JSON row ignored: %s", exc)
                    except Exception as exc:
                        logger.error("Error processing line: %s", exc)

                # Process final remaining lines
                if text_batch and (limit is None or total_rows_processed < limit):
                    # Cut batch size if it exceeds remaining limit
                    if limit is not None:
                        remaining = limit - total_rows_processed
                        text_batch = text_batch[:remaining]
                        meta_batch = meta_batch[:remaining]

                    if text_batch:
                        embeddings = self.model.encode(
                            text_batch,
                            batch_size=len(text_batch),
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                        )
                        all_embeddings.append(embeddings)
                        self.metadata_map.extend(meta_batch)
                        total_rows_processed += len(text_batch)

            if not all_embeddings:
                raise ValueError("No valid text embeddings were generated from the corpus database.")

            # Stack batch embeddings into a single dense matrix
            embeddings_matrix = np.vstack(all_embeddings).astype("float32")

            # Cache computed embeddings to disk for future fast loads (if not a limited build)
            if limit is None:
                logger.info("Caching dense embeddings matrix to disk: %s", npy_cache_path)
                np.save(str(npy_cache_path), embeddings_matrix)

        dimension = embeddings_matrix.shape[1]
        logger.info("Generated dense matrix shape: %s", embeddings_matrix.shape)
        logger.info("Initializing FAISS HNSW Flat Index (dimension: %d, Metric: METRIC_INNER_PRODUCT)", dimension)

        # Initialize HNSW index utilizing Inner Product (which behaves as Cosine Similarity over L2 normalized vectors)
        # M = hnsw_m (connections per node), construction parameter is default
        self.index = faiss.IndexHNSWFlat(dimension, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        
        # Configure search graph parameters
        self.index.hnsw.efSearch = 64
        self.index.hnsw.efConstruction = 128

        logger.info("Adding embedding matrix to FAISS index...")
        self.index.add(embeddings_matrix)

        # Write to disk
        faiss.write_index(self.index, str(self.index_path))
        
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata_map, f, indent=2, ensure_ascii=False)

        elapsed = time.perf_counter() - t_start
        logger.info("=" * 70)
        logger.info("  INDEX BUILDING COMPLETE")
        logger.info("  Total rows indexed   : %d", total_rows_processed)
        logger.info("  Time elapsed         : %.1f seconds", elapsed)
        logger.info("  Serialized index to  : %s", self.index_path)
        logger.info("  Serialized metadata  : %s", self.metadata_path)
        logger.info("=" * 70)

    def load_index(self) -> None:
        """
        Load the serialized FAISS HNSW graph index and matching metadata map from disk.
        """
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found at: {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata mapping file not found at: {self.metadata_path}")

        logger.info("Loading FAISS index from: %s", self.index_path)
        self.index = faiss.read_index(str(self.index_path))

        logger.info("Loading metadata maps from: %s", self.metadata_path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata_map = json.load(f)

        logger.info("Index loaded. Total vectors inside: %d", self.index.ntotal)

    def search(
        self,
        query: str,
        top_k: int = 5,
        query_year: int = 2026,
        decay_lambda: float = 0.005,
    ) -> List[Dict[str, Any]]:
        """
        Vectorize a legal query on the fly, execute a fast HNSW graph search,
        apply a chronological decay penalty to the similarity scores, and
        return the top-k nearest neighbors with adjusted similarity scores.

        Args:
            query: The string-based query.
            top_k: Number of nearest matches to return (default 5).
            query_year: Year of the query to evaluate chronological delta (default 2026).
            decay_lambda: Time decay parameter (default 0.005).

        Returns:
            List[Dict[str, Any]]: Structured matches with scores and metadata.
        """
        import re
        t_start = time.perf_counter()
        if self.index is None or not self.metadata_map:
            logger.info("Cold starting / loading index from default location...")
            self.load_index()

        assert self.index is not None, "FAISS index failed to initialize."

        # Vectorize incoming query string and normalize to unit length
        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        # Execute search with candidate expansion to allow decay re-ranking
        candidate_k = max(top_k * 3, 15)
        D, I = self.index.search(query_vector, candidate_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metadata_map):
                continue
            
            meta = self.metadata_map[idx].copy()
            cosine_sim = float(np.clip(score, -1.0, 1.0))
            
            # Extract doc_year from source_pdf or chunk_id
            doc_year = 2024
            source_pdf = meta.get("source_pdf", "")
            match = re.search(r"\b(19\d{2}|20\d{2})\b", source_pdf)
            if match:
                doc_year = int(match.group(1))
            else:
                chunk_id = meta.get("chunk_id", "")
                match = re.search(r"_(19\d{2}|20\d{2})_", chunk_id)
                if match:
                    doc_year = int(match.group(1))
                    
            # Compute temporal decay
            delta_years = max(0, query_year - doc_year)
            adjusted_score = cosine_sim - (decay_lambda * delta_years)
            
            meta["cosine_similarity"] = float(adjusted_score)
            results.append(meta)

        # Sort again by adjusted score descending
        results.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        # Select top_k
        results = results[:top_k]
        
        # Assign ranks
        for rank, meta in enumerate(results, start=1):
            meta["rank"] = rank

        latency_ms = (time.perf_counter() - t_start) * 1000
        logger.info("Query retrieved in %.2f ms | Top result score: %.4f", latency_ms, results[0]["cosine_similarity"] if results else 0.0)
        return results

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 ─ ENTRYPOINT & INTEGRATION VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Initializing search engine test pipeline...")
    
    # 1. Initialize engine
    engine = LegalSearchEngine(index_dir="./LAWdata_Corpus_2024")

    # 2. Check if index files exist; if not, build it.
    if not engine.index_path.exists() or not engine.metadata_path.exists():
        logger.info("Serialized index not found on disk. Building from scratch...")
        engine.build_index()
    else:
        logger.info("Index detected on disk. Loading cached state.")
        engine.load_index()

    # 3. Perform a demonstration search
    demo_query = "What is the disqualification criteria for a female Sarpanch who encroached on government land?"
    logger.info("Executing demonstration semantic search query: '%s'", demo_query)
    
    hits = engine.search(demo_query, top_k=3)
    
    print("\n" + "=" * 80)
    print(f"SEARH HITS FOR: '{demo_query}'")
    print("=" * 80)
    for hit in hits:
        print(f"Rank {hit['rank']} (Score: {hit['cosine_similarity']:.4f})")
        print(f"  Chunk ID   : {hit['chunk_id']}")
        print(f"  Source PDF : {hit['source_pdf']}")
        print(f"  Snippet    : {hit['text_chunk'][:150]}...")
        print("-" * 80)
    print("=" * 80 + "\n")
