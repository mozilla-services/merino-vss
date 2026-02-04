"""
build_index.py - Offline script to build FAISS index artifacts.

=============================================================================
WHAT THIS SCRIPT DOES
=============================================================================

This script is the "offline" or "build time" step of the intent classification
system. It reads labeled training examples and produces FAISS index artifacts
that can be loaded at runtime for fast similarity search.

The workflow is:

    1. Read training examples from a JSONL file (text + intent label pairs)
    2. Load a SentenceTransformer model (e.g., all-MiniLM-L6-v2)
    3. Encode all training texts into vector embeddings
    4. Build FAISS indexes from those embeddings
    5. Save the indexes and metadata to disk

The output artifacts are:

    artifacts/
    ├── exemplars.faiss          # FAISS index with one vector per training example
    ├── exemplars_meta.json      # Metadata mapping vector IDs to intents/texts
    ├── centroids.faiss          # FAISS index with one vector per intent (mean)
    └── centroids_meta.json      # Metadata for centroids

=============================================================================
WHEN TO RUN THIS SCRIPT
=============================================================================

Run this script when:
    - You've added/removed/changed training examples in data/train.jsonl
    - You want to try a different embedding model
    - You want to rebuild from scratch for any reason

This script is NOT run in production. It's a one-time (or periodic) offline
step. The artifacts it produces are what gets deployed to production.

=============================================================================
RUNTIME CHARACTERISTICS
=============================================================================

- This script is SLOW (10-60 seconds depending on training set size and model).
- The slowness is due to encoding all training examples through the transformer.
- This is fine because it only runs offline, not in the request path.

=============================================================================
USAGE
=============================================================================

    # Basic usage (uses defaults)
    uv run python scripts/build_index.py

    # Full usage with all options
    uv run python scripts/build_index.py \
        --train data/train.jsonl \
        --artifacts-dir artifacts \
        --model all-MiniLM-L6-v2 \
        --batch-size 64

=============================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vss.intent_faiss import Embedder, FaissIntentIndex, read_jsonl_examples


def main() -> None:
    """
    Main entry point for the index building script.

    This function:
        1. Parses command-line arguments
        2. Loads training examples
        3. Initializes the embedding model
        4. Builds two types of indexes (exemplars and centroids)
        5. Saves all artifacts to disk

    The two index types serve different purposes:
        - Exemplars: One vector per training example. Richer, more nuanced.
        - Centroids: One vector per intent (the mean). Smaller, faster, simpler.

    We build BOTH so you can experiment with which works better for your use case.
    In practice, exemplars usually performs better, but centroids is a good baseline.
    """

    # -------------------------------------------------------------------------
    # Argument parsing
    # -------------------------------------------------------------------------

    p = argparse.ArgumentParser(
        description="Build FAISS index artifacts from training data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--train",
        default="data/train.jsonl",
        help="Path to training data (JSONL format with 'text' and 'intent' fields)",
    )

    p.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory to write output artifacts",
    )

    p.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help=(
            "SentenceTransformer model name. This determines embedding quality "
            "and dimension. IMPORTANT: Must use the same model at query time!"
        ),
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help=(
            "Batch size for encoding. Higher = faster but more memory. "
            "Reduce if you run out of memory."
        ),
    )

    args = p.parse_args()

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    train_path = args.train
    artifacts_dir = Path(args.artifacts_dir)

    # Create artifacts directory if it doesn't exist
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load training data
    # -------------------------------------------------------------------------
    # The training data is a JSONL file where each line is:
    #   {"text": "aapl stock price", "intent": "finance"}
    #
    # You should have multiple examples per intent to capture the diversity
    # of how users might phrase queries. More examples = better coverage,
    # but also larger index and longer build time.

    examples = read_jsonl_examples(train_path)
    print(f"Loaded {len(examples)} training examples from {train_path}")

    # -------------------------------------------------------------------------
    # Initialize embedder
    # -------------------------------------------------------------------------
    # This loads the transformer model. It's slow (~2-5 seconds) because it
    # downloads/loads model weights. In production (Merino), this happens
    # once at startup, not per-request.
    #
    # CRITICAL: The model used here MUST match the model used at query time.
    # If you build with "all-MiniLM-L6-v2" but query with "all-mpnet-base-v2",
    # the embeddings will be incompatible and search results will be garbage.

    print(f"Loading embedding model: {args.model}")
    embedder = Embedder(args.model)

    # -------------------------------------------------------------------------
    # Build exemplars index
    # -------------------------------------------------------------------------
    # This creates an index with one vector per training example.
    #
    # How it works:
    #   1. Encode each training text into a 384-dimensional vector
    #   2. Normalize vectors to unit length (for cosine similarity via dot product)
    #   3. Add all vectors to a FAISS IndexFlatIP (exact inner product search)
    #   4. Store metadata mapping vector IDs to intent labels and original text
    #
    # The result is:
    #   - exemplars.faiss: Binary file with the vectors
    #   - exemplars_meta.json: JSON with metadata (texts, intents, model info)

    print("Building exemplars index...")
    ex_index, ex_meta = FaissIntentIndex.build_exemplars(
        examples=examples,
        embedder=embedder,
        batch_size=int(args.batch_size),
    )

    # Wrap in FaissIntentIndex for the save() method
    ex_store = FaissIntentIndex(index=ex_index, meta=ex_meta)

    # Save to disk
    ex_store.save(
        str(artifacts_dir / "exemplars.faiss"),
        str(artifacts_dir / "exemplars_meta.json"),
    )
    print(f"  Saved: {artifacts_dir}/exemplars.faiss")
    print(f"  Saved: {artifacts_dir}/exemplars_meta.json")

    # -------------------------------------------------------------------------
    # Build centroids index
    # -------------------------------------------------------------------------
    # This creates an index with one vector per intent (the centroid/mean).
    #
    # How it works:
    #   1. Group training examples by intent
    #   2. For each intent, encode all examples and compute the mean vector
    #   3. Re-normalize the centroid (mean of unit vectors isn't necessarily unit)
    #   4. Add all centroids to a FAISS IndexFlatIP
    #
    # Why centroids?
    #   - Much smaller index (3 vectors for 3 intents vs 100s for exemplars)
    #   - Faster search (though exemplars is already fast enough)
    #   - Less sensitive to outlier training examples
    #
    # Why NOT centroids?
    #   - Loses nuance: can't capture multiple "modes" within an intent
    #   - Example: if "finance" includes both stock queries AND crypto queries,
    #     the centroid might be somewhere in between, matching neither well
    #
    # In practice, start with exemplars and use centroids as a baseline comparison.

    print("Building centroids index...")
    c_index, c_meta = FaissIntentIndex.build_centroids(
        examples=examples,
        embedder=embedder,
        batch_size=int(args.batch_size),
    )

    c_store = FaissIntentIndex(index=c_index, meta=c_meta)
    c_store.save(
        str(artifacts_dir / "centroids.faiss"),
        str(artifacts_dir / "centroids_meta.json"),
    )
    print(f"  Saved: {artifacts_dir}/centroids.faiss")
    print(f"  Saved: {artifacts_dir}/centroids_meta.json")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    intents = sorted(set(e.intent for e in examples))

    print("")
    print("=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"Model:      {args.model}")
    print(f"Intents:    {intents}")
    print(f"Examples:   {len(examples)}")
    print(f"Artifacts:  {artifacts_dir}/")
    print("")
    print("Next steps:")
    print("  1. Run benchmark:  uv run python scripts/bench.py")
    print("  2. Try a query:    uv run python scripts/query.py 'aapl stock price'")
    print("")


if __name__ == "__main__":
    main()
