"""
vss - Vector Similarity Search module for intent classification.

=============================================================================
PACKAGE OVERVIEW
=============================================================================

This package implements a FAISS-based intent classification system for Merino.
The goal is to classify user search queries (like "aapl stock price") into
intents (like "finance") so Merino can route them to specialized providers.

The system uses semantic similarity: queries are converted to vector embeddings,
and we find the most similar training examples to determine the intent.

=============================================================================
MODULE STRUCTURE
=============================================================================

vss/
├── __init__.py        ← You are here. Package initialization.
├── intent_faiss.py    ← Core classification logic (Embedder, Index, Classifier)
└── bench.py           ← Benchmarking utilities (metrics, latency analysis)

HOW THE MODULES FIT TOGETHER:
-----------------------------

1. intent_faiss.py is the core. It contains:
   - Embedder: Wraps SentenceTransformer for text → vector encoding
   - FaissIntentIndex: Wraps FAISS index + metadata for similarity search
   - IntentClassifier: Combines embedder + index + decision logic

2. bench.py is for evaluation. It contains:
   - compute_metrics(): Calculate precision, recall, false positive rate
   - summarize_latency(): Calculate p50/p95/p99 latency percentiles

=============================================================================
TYPICAL USAGE
=============================================================================

Building an index (offline):
----------------------------
    from vss.intent_faiss import Embedder, FaissIntentIndex, read_jsonl_examples

    examples = read_jsonl_examples("data/train.jsonl")
    embedder = Embedder("all-MiniLM-L6-v2")
    index, meta = FaissIntentIndex.build_exemplars(examples, embedder, batch_size=64)
    store = FaissIntentIndex(index=index, meta=meta)
    store.save("artifacts/exemplars.faiss", "artifacts/exemplars_meta.json")

Classifying queries (online):
-----------------------------
    from vss.intent_faiss import Embedder, FaissIntentIndex, IntentClassifier

    store = FaissIntentIndex.load("artifacts/exemplars.faiss", "artifacts/exemplars_meta.json")
    embedder = Embedder("all-MiniLM-L6-v2")
    clf = IntentClassifier(index=store, embedder=embedder)

    result = clf.classify("aapl stock price")
    print(result.predicted_intent)  # "finance"
    print(result.confidence)        # 0.87
    print(result.abstained)         # False

Benchmarking:
-------------
    from vss.bench import compute_metrics, summarize_latency

    # After running classifier on test set...
    metrics = compute_metrics(truth=truth_labels, pred=pred_labels, positive_intents=["finance", "flights", "sports"])
    latency = summarize_latency(timings)

    print(metrics.overall["provider_fp_rate"])  # Should be LOW!
    print(latency["encode_p95_ms"])             # ~10ms typical

=============================================================================
FOR MERINO INTEGRATION
=============================================================================

When integrating into Merino, this package should be:
    1. Copied into merino/providers/suggest/intent/ (or similar)
    2. Index artifacts placed in an accessible location (baked into image or fetched)
    3. Classifier loaded once at startup (in provider's initialize() method)
    4. classify() called per-request (fast path: ~7ms encoding + ~0.01ms search)

See the README or architecture docs for detailed integration guidance.

=============================================================================
"""

# Import submodules so they're available in the package namespace
from vss import bench as bench
from vss import intent_faiss as intent_faiss

# Public API: what gets exported when you do "from vss import *"
# In practice, you'll usually import specific classes directly:
#   from vss.intent_faiss import IntentClassifier, Embedder, ...
#   from vss.bench import compute_metrics, summarize_latency
__all__ = ["intent_faiss", "bench"]
