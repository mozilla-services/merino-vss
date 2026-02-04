"""
intent_faiss.py - Core module for FAISS-based intent classification.

=============================================================================
OVERVIEW
=============================================================================

This module implements a vector-similarity-based intent classifier for Merino.
The core idea is simple:

    1. We have labeled example queries (e.g., "aapl stock price" → "finance")
    2. We convert these examples into vector embeddings using a transformer model
    3. We store these embeddings in a FAISS index for fast similarity search
    4. At query time, we embed the user's query and find the nearest neighbors
    5. We aggregate neighbor votes to predict the intent (or abstain if uncertain)

This approach is called "semantic search" or "vector similarity search" because
we're matching based on meaning, not keywords. "plane tickets to NYC" will match
"flights from sfo to jfk" even though they share no words.

=============================================================================
KEY ARCHITECTURAL DECISIONS
=============================================================================

1. WHY FAISS (not a vector database)?
   - FAISS is a library, not a service. No network calls, no infrastructure.
   - For our scale (hundreds to thousands of exemplars), FAISS is overkill-fast.
   - Search latency is ~0.01ms. The bottleneck is embedding, not search.
   - If we ever need to scale to millions of vectors, FAISS supports approximate
     indexes that trade accuracy for speed. We're using exact search for now.

2. WHY SENTENCE-TRANSFORMERS?
   - It's the standard library for text embeddings in Python.
   - Pre-trained models like "all-MiniLM-L6-v2" work well out of the box.
   - We can swap models easily (e.g., multilingual models for i18n).
   - If needed, we can fine-tune on our own data later.

3. WHY L2 NORMALIZATION + INNER PRODUCT (not cosine directly)?
   - Cosine similarity = dot product when vectors are unit length.
   - FAISS's IndexFlatIP (inner product) is faster than computing cosine.
   - We normalize all vectors to unit length at encoding time.
   - This is a standard trick in production vector search systems.

4. WHY "ABSTAIN" AS A FIRST-CLASS CONCEPT?
   - For Merino routing, false positives are worse than false negatives.
   - If we wrongly route a "weather in paris" query to flights, bad UX.
   - Better to abstain (return "none") when confidence is low.
   - The thresholds (min_top_score, min_margin, min_confidence) control this.

5. WHY TWO INDEX TYPES (exemplars vs centroids)?
   - Exemplars: Store every training example. Rich, but larger index.
   - Centroids: Store one vector per intent (the mean). Tiny, but less nuanced.
   - Centroids can fail on "multi-modal" intents (e.g., finance = stocks OR crypto).
   - Start with exemplars; centroids are useful for baseline comparison.

=============================================================================
RUNTIME CHARACTERISTICS (what to expect in production)
=============================================================================

For all-MiniLM-L6-v2 with ~100 exemplars:
    - Model load time: ~2-5 seconds (one-time, at startup)
    - Embedding latency: ~5-10ms per query (this is the bottleneck)
    - Search latency: ~0.01ms per query (FAISS is very fast)
    - Memory: ~1.5KB per vector (384 dims × 4 bytes), plus model (~90MB)

The embedding step is CPU-bound (transformer forward pass). If latency is
critical, consider:
    - Smaller models (e.g., all-MiniLM-L6-v2 is already small)
    - Quantization (ONNX runtime, etc.)
    - Batching multiple queries (if applicable)
    - GPU inference (if available)

=============================================================================
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer

# =============================================================================
# DATA CLASSES
# =============================================================================
# These are simple, immutable containers for passing data around.
# Using @dataclass(frozen=True) makes them hashable and prevents accidental mutation.


@dataclass(frozen=True)
class Example:
    """
    A single labeled training/test example.

    This is what we read from train.jsonl and test.jsonl files.
    Each line in those files looks like: {"text": "aapl stock", "intent": "finance"}
    """

    text: str  # The query text (e.g., "aapl stock price")
    intent: str  # The ground-truth intent label (e.g., "finance")


@dataclass(frozen=True)
class Neighbor:
    """
    A single neighbor returned by FAISS search.

    When we search the index, we get back the k nearest vectors. Each neighbor
    represents one of those vectors, with its metadata looked up from the index.

    This is useful for debugging and explainability - you can see exactly which
    training examples influenced the classification.
    """

    vector_id: int  # The index position in the FAISS index (0, 1, 2, ...)
    score: float  # Similarity score (higher = more similar, range roughly 0-1 for normalized vectors)
    intent: str  # The intent label of this neighbor (looked up from metadata)
    text: str  # The original text of this neighbor (looked up from metadata)


@dataclass(frozen=True)
class Classification:
    """
    The result of classifying a query.

    This contains everything you need to make a routing decision AND to debug
    why the classifier made that decision.

    Key fields for routing:
        - predicted_intent: The intent to route to, or "none" if abstained
        - abstained: True if we decided not to route (low confidence)

    Key fields for debugging/monitoring:
        - confidence: How dominant was the winning intent in the neighborhood?
        - top_score: How similar was the closest neighbor?
        - margin: Gap between best and second-best scores
        - neighbors: The actual neighbors (for manual inspection)
    """

    predicted_intent: str  # "finance", "flights", "sports", or "none"
    confidence: float  # Fraction of neighbor mass belonging to predicted intent (0-1)
    abstained: bool  # Did we abstain from routing?
    top_score: float  # Similarity score of the closest neighbor
    margin: float  # top_score - second_best_score (higher = more decisive)
    neighbors: List[Neighbor]  # The k nearest neighbors (for debugging)


@dataclass(frozen=True)
class TimingsMs:
    """
    Timing breakdown for a single classification call.

    This is critical for performance monitoring. In production, you want to
    track these percentiles (p50, p95, p99) to understand latency distribution.

    Typical values for all-MiniLM-L6-v2:
        - encode_ms: 5-10ms (this dominates)
        - search_ms: 0.01-0.05ms (FAISS is fast)
        - total_ms: ~encode_ms (search is negligible)
    """

    encode_ms: float  # Time to embed the query (transformer forward pass)
    search_ms: float  # Time to search the FAISS index
    total_ms: float  # End-to-end time (includes aggregation logic)


# =============================================================================
# DATA LOADING
# =============================================================================


def read_jsonl_examples(path: str) -> List[Example]:
    """
    Read labeled examples from a JSONL file.

    JSONL (JSON Lines) format: one JSON object per line. Example:
        {"text": "aapl stock price", "intent": "finance"}
        {"text": "flights to paris", "intent": "flights"}

    This format is convenient because:
        - Easy to append new examples (just add a line)
        - Easy to inspect/edit with standard tools (grep, head, etc.)
        - Streams well (don't need to load entire file to parse)

    Args:
        path: Path to the JSONL file

    Returns:
        List of Example objects
    """
    out: List[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue  # Skip empty lines
            obj = json.loads(line)
            out.append(Example(text=str(obj["text"]), intent=str(obj["intent"])))
    return out


# =============================================================================
# VECTOR NORMALIZATION
# =============================================================================


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    """
    Normalize each row of a matrix to unit length (L2 norm = 1).

    WHY THIS MATTERS:
    -----------------
    Cosine similarity between vectors a and b is:
        cos(a, b) = (a · b) / (||a|| * ||b||)

    If both vectors are already unit length (||a|| = ||b|| = 1), then:
        cos(a, b) = a · b  (just the dot product!)

    FAISS's IndexFlatIP computes dot products (inner products). By normalizing
    our vectors, we get cosine similarity "for free" with better performance.

    This is a standard trick in production vector search systems.

    Args:
        x: A 2D numpy array of shape (n_vectors, embedding_dim)

    Returns:
        The same array with each row normalized to unit length
    """
    # Compute the L2 norm of each row
    norms = np.linalg.norm(x, axis=1, keepdims=True)

    # Avoid division by zero for zero vectors (shouldn't happen, but defensive)
    norms = np.maximum(norms, 1e-12)

    return x / norms


# =============================================================================
# EMBEDDER
# =============================================================================


class Embedder:
    """
    Wrapper around SentenceTransformer for encoding text into vectors.

    This class exists to:
        1. Encapsulate model loading and configuration
        2. Ensure consistent normalization (all vectors are unit length)
        3. Provide a clean interface for both batch encoding and single-query encoding

    ABOUT THE MODEL (all-MiniLM-L6-v2):
    -----------------------------------
    - Architecture: 6-layer MiniLM (distilled from larger models)
    - Embedding dimension: 384
    - Trained on: 1B+ sentence pairs for semantic similarity
    - Size: ~90MB
    - Speed: Fast enough for real-time inference on CPU

    You can swap this for other models:
        - "all-mpnet-base-v2": Better quality, slower, 768 dims
        - "paraphrase-multilingual-MiniLM-L12-v2": Multilingual support
        - A fine-tuned model on your own data

    IMPORTANT: The model used at query time MUST match the model used to build
    the index. Embeddings from different models are NOT compatible.
    """

    def __init__(self, model_name: str) -> None:
        """
        Initialize the embedder by loading the SentenceTransformer model.

        This is SLOW (~2-5 seconds) because it loads the model weights.
        In production, do this once at startup, not per-request.

        Args:
            model_name: HuggingFace model name (e.g., "all-MiniLM-L6-v2")
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode a batch of texts into normalized embeddings.

        Used during index building to encode all training exemplars.

        Args:
            texts: List of strings to encode
            batch_size: How many texts to encode at once (memory vs speed tradeoff)

        Returns:
            numpy array of shape (len(texts), embedding_dim), dtype float32
            Each row is a unit-length embedding vector.
        """
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,  # Quiet for production use
        )
        arr = np.asarray(emb, dtype="float32")
        return l2_normalize_rows(arr)

    def encode_query(self, text: str) -> np.ndarray:
        """
        Encode a single query into a normalized embedding.

        Used at inference time to encode the user's search query.

        Note: We still return a 2D array of shape (1, embedding_dim) because
        FAISS expects 2D input for search.

        Args:
            text: The query string

        Returns:
            numpy array of shape (1, embedding_dim), dtype float32
        """
        emb = self.model.encode(
            [text],
            batch_size=1,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        arr = np.asarray(emb, dtype="float32")
        arr = l2_normalize_rows(arr)
        return arr


# =============================================================================
# FAISS INDEX WRAPPER
# =============================================================================


class FaissIntentIndex:
    """
    A minimal "vector database" for intent classification.

    This class bundles:
        1. A FAISS index (the actual vectors, stored in *.faiss files)
        2. Metadata (mapping vector IDs to intent labels and original text)

    WHY TWO FILES?
    --------------
    FAISS only stores vectors. It returns integer IDs when you search.
    We need a separate metadata file to look up what each ID means:
        - ID 0 → intent="finance", text="aapl stock price"
        - ID 1 → intent="finance", text="msft stock"
        - etc.

    The metadata is stored as JSON for easy inspection and debugging.

    FAISS INDEX TYPES:
    ------------------
    We use IndexFlatIP (Flat = exact search, IP = Inner Product).

    - "Flat" means we compare the query against ALL vectors (no approximation).
      This is fine for small datasets (<100K vectors). For larger datasets,
      you'd use approximate indexes like IVF or HNSW.

    - "IP" means Inner Product (dot product). Since our vectors are normalized,
      this gives us cosine similarity.

    Alternative: IndexFlatL2 uses Euclidean distance. For normalized vectors,
    L2 distance and cosine similarity give equivalent rankings, but the scores
    are different (L2 is a distance, lower is better; IP is a similarity, higher is better).
    """

    def __init__(self, index: faiss.Index, meta: Dict[str, Any]) -> None:
        """
        Initialize from an existing FAISS index and metadata dict.

        Typically you don't call this directly. Use:
            - build_exemplars() / build_centroids() to create a new index
            - load() to load an existing index from disk

        Args:
            index: A FAISS index object
            meta: Metadata dict containing texts, intents, model info
        """
        self.index = index
        self.meta = meta

        # Extract commonly-used fields for fast access
        self.texts: List[str] = list(meta["texts"])
        self.intents: List[str] = list(meta["intents"])
        self.model_name: str = str(meta["model_name"])
        self.index_kind: str = str(meta["index_kind"])

        # Sanity check: texts and intents must align with index size
        if len(self.texts) != len(self.intents):
            raise ValueError("meta.json invalid: texts and intents length mismatch")

    # -------------------------------------------------------------------------
    # Index Building (offline, run once)
    # -------------------------------------------------------------------------

    @staticmethod
    def build_exemplars(
        examples: List[Example],
        embedder: Embedder,
        batch_size: int,
    ) -> Tuple[faiss.Index, Dict[str, Any]]:
        """
        Build an "exemplars" index: one vector per training example.

        This is the recommended starting point. Each training example becomes
        a vector in the index. At query time, we find the nearest examples
        and aggregate their intent labels.

        TRADEOFFS:
        ----------
        Pros:
            - Rich coverage of query variations
            - Explainable: neighbors are real examples you can inspect
            - Easy to update: add more examples, rebuild

        Cons:
            - Larger index (one vector per example)
            - Quality depends on having diverse, representative examples

        Args:
            examples: List of labeled Example objects
            embedder: An Embedder instance (must be the same model used at query time!)
            batch_size: Batch size for encoding (memory vs speed)

        Returns:
            Tuple of (faiss_index, metadata_dict)
        """
        texts = [e.text for e in examples]
        intents = [e.intent for e in examples]

        # Encode all examples into vectors
        # Shape: (n_examples, embedding_dim), e.g., (100, 384)
        vecs = embedder.encode_texts(texts, batch_size=batch_size)
        d = int(vecs.shape[1])  # embedding dimension

        # Create a FAISS index for inner product search
        # IndexFlatIP = exact search using dot product
        index = faiss.IndexFlatIP(d)

        # Add all vectors to the index
        # After this, searching will compare against all these vectors
        index.add(vecs)  # type: ignore[call-arg]

        # Store metadata so we can look up intent/text for each vector ID
        meta: Dict[str, Any] = {
            "model_name": embedder.model_name,
            "built_at_unix": time.time(),
            "dim": d,
            "index_kind": "exemplars",
            "faiss_index_type": "IndexFlatIP",
            "normalized": True,  # Important: vectors are unit length
            "texts": texts,  # texts[i] is the original text for vector i
            "intents": intents,  # intents[i] is the intent label for vector i
        }
        return index, meta

    @staticmethod
    def build_centroids(
        examples: List[Example],
        embedder: Embedder,
        batch_size: int,
    ) -> Tuple[faiss.Index, Dict[str, Any]]:
        """
        Build a "centroids" index: one vector per intent (the mean of all examples).

        This is an alternative to exemplars. Instead of storing every example,
        we compute the average embedding for each intent and store just that.

        TRADEOFFS:
        ----------
        Pros:
            - Tiny index (one vector per intent, e.g., 3 vectors for 3 intents)
            - Very fast (but exemplars is already fast enough)
            - Stable: not sensitive to outlier examples

        Cons:
            - Loses nuance: can't distinguish sub-types within an intent
            - Fails on "multi-modal" intents (e.g., if finance = stocks AND crypto,
              the centroid might be somewhere in between, matching neither well)
            - Harder to debug: neighbors are centroids, not real examples

        WHEN TO USE:
        ------------
        - As a baseline to compare against exemplars
        - If you have very few intents and well-separated clusters
        - If you need an extremely small index (e.g., edge deployment)

        Args:
            examples: List of labeled Example objects
            embedder: An Embedder instance
            batch_size: Batch size for encoding

        Returns:
            Tuple of (faiss_index, metadata_dict)
        """
        # Group examples by intent
        by_intent: Dict[str, List[str]] = {}
        for e in examples:
            by_intent.setdefault(e.intent, []).append(e.text)

        intents_sorted = sorted(by_intent.keys())
        centroid_texts: List[str] = []
        centroid_intents: List[str] = []
        centroid_vecs: List[np.ndarray] = []

        for intent in intents_sorted:
            texts = by_intent[intent]

            # Encode all examples for this intent
            vecs = embedder.encode_texts(texts, batch_size=batch_size)

            # Compute the centroid (mean of all vectors)
            # keepdims=True keeps shape as (1, dim) instead of (dim,)
            centroid = np.mean(vecs, axis=0, keepdims=True).astype("float32")

            # Re-normalize after averaging (mean of unit vectors isn't necessarily unit)
            centroid = l2_normalize_rows(centroid)

            centroid_vecs.append(centroid)
            centroid_intents.append(intent)
            # Store a descriptive text for debugging
            centroid_texts.append(f"centroid:{intent} ({len(texts)} examples)")

        # Stack all centroids into a single matrix
        mat = np.vstack(centroid_vecs)
        d = int(mat.shape[1])

        index = faiss.IndexFlatIP(d)
        index.add(mat)  # type: ignore[call-arg]

        meta: Dict[str, Any] = {
            "model_name": embedder.model_name,
            "built_at_unix": time.time(),
            "dim": d,
            "index_kind": "centroids",
            "faiss_index_type": "IndexFlatIP",
            "normalized": True,
            "texts": centroid_texts,
            "intents": centroid_intents,
        }
        return index, meta

    # -------------------------------------------------------------------------
    # Persistence (save/load to disk)
    # -------------------------------------------------------------------------

    def save(self, index_path: str, meta_path: str) -> None:
        """
        Save the index and metadata to disk.

        This produces two files:
            - index_path (e.g., "artifacts/exemplars.faiss"): Binary FAISS index
            - meta_path (e.g., "artifacts/exemplars_meta.json"): JSON metadata

        The FAISS file is binary and not human-readable. The metadata JSON is
        human-readable and useful for debugging.

        Args:
            index_path: Where to save the FAISS index
            meta_path: Where to save the metadata JSON
        """
        # Create directories if they don't exist
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(meta_path).parent.mkdir(parents=True, exist_ok=True)

        # FAISS has its own serialization format
        faiss.write_index(self.index, index_path)

        # Metadata is plain JSON
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)

    @staticmethod
    def load(index_path: str, meta_path: str) -> "FaissIntentIndex":
        """
        Load an index and metadata from disk.

        Used at Merino startup to load pre-built artifacts.

        Args:
            index_path: Path to the FAISS index file
            meta_path: Path to the metadata JSON file

        Returns:
            A FaissIntentIndex instance ready for searching
        """
        index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return FaissIntentIndex(index=index, meta=meta)


# =============================================================================
# INTENT CLASSIFIER
# =============================================================================


class IntentClassifier:
    """
    The main classification engine: takes a query, returns an intent (or abstains).

    This class combines:
        1. An Embedder (to convert queries to vectors)
        2. A FaissIntentIndex (to find similar vectors)
        3. Decision logic (to aggregate neighbors and decide intent)

    THE DECISION LOGIC (this is where the magic happens):
    -----------------------------------------------------

    1. Embed the query using the same model used to build the index.

    2. Search FAISS for the k nearest neighbors.

    3. Aggregate neighbor scores by intent:
       - For each neighbor, add its similarity score to its intent's total
       - The "winning" intent is the one with the highest total score
       - Confidence = winning_intent_score / total_score (what fraction of
         the neighborhood belongs to the predicted intent)

    4. Apply abstain thresholds:
       - min_top_score: Is the best match strong enough?
       - min_margin: Is the best match clearly better than the second?
       - min_confidence: Does one intent dominate the neighborhood?

       If ANY threshold fails, we abstain (return "none").

    WHY THESE THREE THRESHOLDS?
    ---------------------------

    - top_score catches "out of distribution" queries. If the query is unlike
      anything in the training set, even the best match will have low similarity.
      Example: "how to boil eggs" might match finance queries with score 0.2.

    - margin catches "ambiguous" queries. If two intents are nearly tied,
      we shouldn't confidently pick one.
      Example: "airport" might match both flights (0.7) and weather (0.68).

    - confidence catches "scattered" neighborhoods. If the top-k neighbors
      are split across multiple intents, we shouldn't trust any single one.
      Example: neighbors are [finance, flights, sports, finance, flights, ...]

    In practice, you tune these thresholds based on your precision/recall
    requirements. Higher thresholds = more abstaining = higher precision,
    lower recall.

    ABOUT k (number of neighbors):
    ------------------------------
    k=8 is a reasonable default. Too small and you're sensitive to noise.
    Too large and distant neighbors dilute the signal. Experiment to find
    what works for your data.
    """

    def __init__(
        self,
        index: FaissIntentIndex,
        embedder: Embedder,
        k: int = 8,
        min_top_score: float = 0.45,
        min_margin: float = 0.05,
        min_confidence: float = 0.60,
    ) -> None:
        """
        Initialize the classifier.

        Args:
            index: A FaissIntentIndex (the "database" of exemplars/centroids)
            embedder: An Embedder (MUST be the same model used to build the index!)
            k: Number of neighbors to retrieve
            min_top_score: Minimum similarity for the best match (0-1)
            min_margin: Minimum gap between best and second-best score
            min_confidence: Minimum fraction of neighborhood for winning intent
        """
        self.index = index
        self.embedder = embedder
        self.k = int(k)

        # Threshold configuration
        # These are the "knobs" you tune for precision/recall tradeoff
        self.min_top_score = float(min_top_score)
        self.min_margin = float(min_margin)
        self.min_confidence = float(min_confidence)

    @staticmethod
    def normalize_query(text: str) -> str:
        """
        Normalize a query string before classification.

        Current normalization:
            - Strip leading/trailing whitespace
            - Collapse multiple spaces into one

        This ensures "  aapl   stock " matches "aapl stock".

        Note: We intentionally DON'T lowercase here because the embedding model
        may encode case information. If you want case-insensitive matching,
        you could add .lower() here, but test the impact on quality first.

        Args:
            text: Raw query string

        Returns:
            Normalized query string
        """
        return " ".join((text or "").strip().split())

    def classify(self, query: str) -> Classification:
        """
        Classify a query's intent.

        This is the simple interface when you don't need timing info.

        Args:
            query: The user's search query

        Returns:
            A Classification object with predicted intent and confidence
        """
        cls, _ = self.classify_with_timings(query)
        return cls

    def classify_with_timings(self, query: str) -> Tuple[Classification, TimingsMs]:
        """
        Classify a query's intent and return timing breakdown.

        This is the full interface for production use where you want to
        monitor latency.

        THE ALGORITHM:
        --------------
        1. Normalize the query
        2. Embed the query (this is the slow part: ~5-10ms)
        3. Search FAISS for k nearest neighbors (~0.01ms)
        4. Build neighbor list with metadata lookup
        5. Aggregate scores by intent (weighted voting)
        6. Apply abstain thresholds
        7. Return classification and timings

        Args:
            query: The user's search query

        Returns:
            Tuple of (Classification, TimingsMs)
        """
        # --- Normalize ---
        q = self.normalize_query(query)
        if not q:
            # Empty query: return immediately with abstain
            empty = Classification(
                predicted_intent="none",
                confidence=0.0,
                abstained=True,
                top_score=0.0,
                margin=0.0,
                neighbors=[],
            )
            zero = TimingsMs(encode_ms=0.0, search_ms=0.0, total_ms=0.0)
            return empty, zero

        # --- Timing setup ---
        # Using perf_counter_ns for high-resolution timing
        t0 = time.perf_counter_ns()
        t1 = t0

        # --- Embed the query ---
        # This is the expensive step (~5-10ms for MiniLM)
        q_vec = self.embedder.encode_query(q)
        t2 = time.perf_counter_ns()

        # --- Search FAISS ---
        # Returns:
        #   scores: shape (1, k), similarity scores (higher = more similar)
        #   ids: shape (1, k), vector IDs (indexes into the FAISS index)
        # Both are 2D because FAISS supports batch queries; we only have 1 query.
        scores, ids = self.index.index.search(q_vec, self.k)  # type: ignore[call-arg]
        t3 = time.perf_counter_ns()

        # Convert to Python lists (FAISS returns numpy arrays)
        score_list = scores[0].tolist()
        id_list = ids[0].tolist()

        # --- Build neighbor list ---
        # Look up metadata for each neighbor
        neighbors: List[Neighbor] = []
        for s, i in zip(score_list, id_list):
            # FAISS returns -1 for "no result" (happens if k > index size)
            if i < 0:
                continue
            neighbors.append(
                Neighbor(
                    vector_id=int(i),
                    score=float(s),
                    intent=self.index.intents[int(i)],
                    text=self.index.texts[int(i)],
                )
            )

        # Edge case: no valid neighbors
        if not neighbors:
            none = Classification(
                predicted_intent="none",
                confidence=0.0,
                abstained=True,
                top_score=0.0,
                margin=0.0,
                neighbors=[],
            )
            timings = TimingsMs(
                encode_ms=(t2 - t1) / 1e6,
                search_ms=(t3 - t2) / 1e6,
                total_ms=(t3 - t0) / 1e6,
            )
            return none, timings

        # --- Compute decision signals ---

        # Top score: similarity of the closest neighbor
        top_score = neighbors[0].score

        # Margin: gap between best and second-best
        # If there's only one neighbor, use -1 as a sentinel (will likely fail margin check)
        second_score = neighbors[1].score if len(neighbors) > 1 else -1.0
        margin = float(top_score - second_score)

        # --- Aggregate scores by intent (weighted voting) ---
        # Each neighbor votes for its intent with weight = similarity score
        # This is like k-NN classification but with weighted votes instead of uniform votes.
        #
        # Example with k=5 neighbors:
        #   neighbor 0: intent=finance, score=0.85
        #   neighbor 1: intent=finance, score=0.78
        #   neighbor 2: intent=flights, score=0.72
        #   neighbor 3: intent=finance, score=0.70
        #   neighbor 4: intent=sports,  score=0.65
        #
        # Aggregated:
        #   finance: 0.85 + 0.78 + 0.70 = 2.33
        #   flights: 0.72
        #   sports:  0.65
        #   total:   3.70
        #
        # Winner: finance
        # Confidence: 2.33 / 3.70 = 0.63 (63% of the "mass" is finance)
        #
        by_intent: Dict[str, float] = {}
        total_mass = 0.0
        for n in neighbors:
            # Clamp negative scores to 0 (shouldn't happen with normalized vectors, but defensive)
            mass = max(0.0, float(n.score))
            by_intent[n.intent] = by_intent.get(n.intent, 0.0) + mass
            total_mass += mass

        # --- Determine winner and confidence ---
        if total_mass <= 0.0:
            # Edge case: all scores were zero or negative (very unlikely)
            predicted = "none"
            confidence = 0.0
        else:
            # Winner is the intent with the highest total mass
            predicted = max(by_intent, key=lambda intent: by_intent[intent])
            # Confidence is what fraction of the neighborhood belongs to the winner
            confidence = float(by_intent[predicted] / total_mass)

        # --- Apply abstain thresholds ---
        # This is the key risk-management logic. We abstain if ANY threshold fails.
        #
        # In Merino's context, abstaining means we DON'T route to a specialized
        # provider. The query falls through to the default search engine.
        # This is safe: false negatives (not routing when we should) are better
        # than false positives (routing to the wrong provider).
        #
        abstain = (
            top_score < self.min_top_score  # Best match not strong enough?
            or margin < self.min_margin  # Best match not clearly better than second?
            or confidence < self.min_confidence  # Neighborhood too scattered?
        )

        # Final prediction: the winner, or "none" if we abstained
        predicted_intent = "none" if abstain else predicted

        # --- Build result ---
        cls = Classification(
            predicted_intent=predicted_intent,
            confidence=confidence,
            abstained=abstain,
            top_score=float(top_score),
            margin=float(margin),
            neighbors=neighbors,  # Include for debugging/explainability
        )

        # --- Record timing ---
        t4 = time.perf_counter_ns()
        timings = TimingsMs(
            encode_ms=(t2 - t1) / 1e6,  # Convert nanoseconds to milliseconds
            search_ms=(t3 - t2) / 1e6,
            total_ms=(t4 - t0) / 1e6,
        )

        return cls, timings
