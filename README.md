# merino-vss

**Vector Similarity Search for Intent Classification in Merino**

This repository contains a spike/prototype for using FAISS-based semantic search to classify user query intents in Merino. The goal is to route queries like "aapl stock price" to specialized providers (finance, flights, sports) instead of generic web search.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [How It Works](#how-it-works)
5. [Project Structure](#project-structure)
6. [Usage](#usage)
7. [Benchmarking](#benchmarking)
8. [Tuning Thresholds](#tuning-thresholds)
9. [Merino Integration](#merino-integration)
10. [Key Design Decisions](#key-design-decisions)
11. [FAQ](#faq)

---

## Overview

### The Problem

Merino is Firefox's suggestion service. When a user types in the address bar, we want to:

1. **Recognize the intent** — Is this a finance/flights/sports query?
2. **Route appropriately** — Send to specialized providers for better suggestions
3. **Be conservative** — If uncertain, don't route (abstain)

### The Solution

We use **semantic similarity search**:

1. Convert queries to vector embeddings using a transformer model
2. Compare against a database of labeled example queries
3. Classify based on nearest neighbors (like k-NN, but with embeddings)
4. Abstain when confidence is low (critical for production safety)

### Why This Approach?

- **Semantic matching**: "plane tickets to NYC" matches "flights from sfo to jfk" even without shared keywords
- **Simple to bootstrap**: Just provide labeled examples, no model training required
- **Explainable**: Can inspect which examples influenced a classification
- **Fast**: ~7ms per classification on CPU

---

## Quick Start

```bash
# Install dependencies
uv sync --group dev

# Build the FAISS index from training examples
uv run python scripts/build_index.py

# Run benchmarks against test set
uv run python scripts/bench.py

# Try classifying a query
uv run python scripts/query.py "aapl stock price"
uv run python scripts/query.py "flights to paris"
uv run python scripts/query.py "how to boil eggs"  # Should abstain!
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           OFFLINE (Build Time)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   data/train.jsonl                                                      │
│   (labeled examples)                                                    │
│         │                                                               │
│         ▼                                                               │
│   ┌─────────────────┐                                                   │
│   │ SentenceTransf. │  "aapl stock" → [0.12, -0.34, ..., 0.78]         │
│   │ (Embedder)      │  (384-dimensional vector)                         │
│   └─────────────────┘                                                   │
│         │                                                               │
│         ▼                                                               │
│   ┌─────────────────┐                                                   │
│   │   FAISS Index   │  Store all vectors for fast retrieval             │
│   │ (IndexFlatIP)   │                                                   │
│   └─────────────────┘                                                   │
│         │                                                               │
│         ▼                                                               │
│   artifacts/                                                            │
│   ├── exemplars.faiss       (the vectors)                               │
│   └── exemplars_meta.json   (text + intent for each vector)             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           ONLINE (Query Time)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   User query: "plane tickets to nyc"                                    │
│         │                                                               │
│         ▼                                                               │
│   ┌─────────────────┐                                                   │
│   │ SentenceTransf. │  Same model as build time!                        │
│   │ (Embedder)      │  → [0.11, -0.32, ..., 0.77]                       │
│   └─────────────────┘                                                   │
│         │                                                               │
│         ▼                                                               │
│   ┌─────────────────┐                                                   │
│   │  FAISS Search   │  Find k=8 nearest neighbors                       │
│   └─────────────────┘                                                   │
│         │                                                               │
│         ▼                                                               │
│   ┌─────────────────┐                                                   │
│   │ Decision Logic  │  Aggregate neighbor votes                         │
│   │                 │  Apply abstain thresholds                         │
│   └─────────────────┘                                                   │
│         │                                                               │
│         ▼                                                               │
│   Output: intent="flights", confidence=0.85                             │
│       OR: intent="none" (abstained)                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## How It Works

### 1. Embedding

Text is converted to vectors using a pre-trained transformer model (SentenceTransformers):

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("aapl stock price")  # → 384-dim vector
```

**Key insight**: Similar meanings produce similar vectors, even with different words.

### 2. Similarity Search

We use FAISS (Facebook AI Similarity Search) to efficiently find nearest neighbors:

```python
import faiss

# Build index (offline)
index = faiss.IndexFlatIP(384)  # Inner Product for cosine similarity
index.add(normalized_embeddings)

# Search (online)
scores, ids = index.search(query_embedding, k=8)  # top-8 neighbors
```

### 3. Classification Logic

The classifier aggregates neighbor votes and applies safety thresholds:

```python
# Weighted voting: each neighbor votes for its intent
by_intent = {"finance": 2.33, "flights": 0.72, "sports": 0.65}
winner = "finance"
confidence = 2.33 / 3.70 = 0.63

# Abstain if ANY threshold fails:
abstain = (
    top_score < 0.45      # Best match not strong enough?
    or margin < 0.05       # Best not clearly better than second?
    or confidence < 0.60   # No intent dominates neighborhood?
)
```

### 4. The Abstain Decision

**This is critical for production safety.**

For Merino routing, false positives are worse than false negatives:
- **Bad**: Routing "python list sort" to flights provider
- **Acceptable**: Not routing a legitimate flights query (user still gets web results)

The thresholds control precision/recall tradeoff:
- Higher thresholds → More abstaining → Higher precision, lower recall
- Lower thresholds → Less abstaining → Lower precision, higher recall

---

## Project Structure

```
merino-vss/
├── vss/                          # Core Python package
│   ├── __init__.py               # Package exports
│   ├── intent_faiss.py           # Core: Embedder, FaissIntentIndex, IntentClassifier
│   └── bench.py                  # Benchmarking: compute_metrics, summarize_latency
│
├── scripts/                      # CLI tools
│   ├── build_index.py            # Build FAISS index from training data
│   ├── bench.py                  # Run quality + latency benchmarks
│   └── query.py                  # Classify a single query (debugging)
│
├── data/                         # Training and test data
│   ├── train.jsonl               # Labeled training examples
│   └── test.jsonl                # Labeled test examples (with "none" class!)
│
├── artifacts/                    # Generated index files (don't commit large files)
│   ├── exemplars.faiss           # FAISS index (one vector per example)
│   ├── exemplars_meta.json       # Metadata (text + intent mapping)
│   ├── centroids.faiss           # FAISS index (one vector per intent)
│   └── centroids_meta.json       # Metadata for centroids
│
├── main.py                       # Demo/exploration script
├── pyproject.toml                # Dependencies and project config
└── README.md                     # You are here!
```

---

## Usage

### Building the Index

```bash
# Default: uses data/train.jsonl and all-MiniLM-L6-v2
uv run python scripts/build_index.py

# Custom options
uv run python scripts/build_index.py \
    --train data/train.jsonl \
    --artifacts-dir artifacts \
    --model all-MiniLM-L6-v2 \
    --batch-size 64
```

**Output:**
- `artifacts/exemplars.faiss` — One vector per training example
- `artifacts/exemplars_meta.json` — Metadata mapping
- `artifacts/centroids.faiss` — One vector per intent (alternative, smaller)
- `artifacts/centroids_meta.json` — Metadata for centroids

### Classifying Queries

```bash
# Single query with detailed debug output
uv run python scripts/query.py "aapl stock price"

# Try different queries
uv run python scripts/query.py "flights to paris"
uv run python scripts/query.py "lakers game tonight"
uv run python scripts/query.py "how to boil eggs"  # Should abstain!

# Use centroids instead of exemplars
uv run python scripts/query.py "aapl stock" --index-kind centroids

# Experiment with thresholds
uv run python scripts/query.py "aapl stock" --min-top-score 0.60
```

### Python API

```python
from vss.intent_faiss import Embedder, FaissIntentIndex, IntentClassifier

# Load artifacts (do this once at startup)
store = FaissIntentIndex.load(
    "artifacts/exemplars.faiss",
    "artifacts/exemplars_meta.json"
)
embedder = Embedder("all-MiniLM-L6-v2")
clf = IntentClassifier(index=store, embedder=embedder)

# Classify queries (fast path, ~7ms per query)
result = clf.classify("aapl stock price")

print(result.predicted_intent)  # "finance" or "none"
print(result.confidence)        # 0.0 - 1.0
print(result.abstained)         # True/False
print(result.neighbors)         # List of nearest training examples
```

---

## Benchmarking

### Running Benchmarks

```bash
# Default benchmark
uv run python scripts/bench.py

# Full options
uv run python scripts/bench.py \
    --test data/test.jsonl \
    --artifacts-dir artifacts \
    --index-kind exemplars \
    --k 8 \
    --min-top-score 0.45 \
    --min-margin 0.05 \
    --min-confidence 0.60 \
    --json-out results.json
```

### Key Metrics to Watch

| Metric | What It Measures | Target |
|--------|------------------|--------|
| `provider_fp_rate` | How often we incorrectly route "none" queries | < 5% (ideally < 1%) |
| `precision` (per intent) | When we predict X, how often correct? | > 90% |
| `recall` (per intent) | Of actual X queries, how many caught? | Trade off with precision |
| `coverage` | Fraction of queries we route | Depends on use case |
| `encode_p95_ms` | 95th percentile encoding latency | < 20ms |
| `total_p99_ms` | 99th percentile total latency | < 50ms |

### Example Output

```
============================================================
OVERALL METRICS
============================================================
n:               10  (total test examples)
accuracy:        0.8000  (fraction correct)
coverage:        0.4000  (fraction routed)
abstain_rate:    0.6000  (fraction not routed)

>>> SAFETY METRIC (should be LOW):
provider_fp_rate: 0.0000  (0/4 'none' queries incorrectly routed)

============================================================
PER-INTENT METRICS
============================================================

  FINANCE
    precision: 1.0000  (when we predict finance, how often correct?)
    recall:    1.0000  (of actual finance, how many caught?)

  FLIGHTS
    precision: 1.0000
    recall:    0.5000

============================================================
LATENCY METRICS (milliseconds)
============================================================

  Encoding (transformer forward pass - the bottleneck):
    p50:    7.102 ms
    p95:    8.052 ms
    p99:    8.189 ms

  Search (FAISS nearest neighbor - negligible):
    p50:    0.010 ms
```

---

## Tuning Thresholds

The three threshold parameters control precision/recall tradeoff:

### `min_top_score` (default: 0.45)

**What it does**: Rejects queries where even the best match isn't similar enough.

**When it triggers**: "how to boil eggs" might match finance with score 0.25.

**Tuning**:
- Raise to reject more out-of-distribution queries
- Lower if you're missing queries that are clearly in-domain

### `min_margin` (default: 0.05)

**What it does**: Rejects queries where the top matches are too close (ambiguous).

**When it triggers**: "airport" matches both flights (0.70) and weather (0.68).

**Tuning**:
- Raise to reject more ambiguous queries
- Lower if you're confident in your intent separation

### `min_confidence` (default: 0.60)

**What it does**: Rejects queries where no intent dominates the neighborhood.

**When it triggers**: Top-8 neighbors are scattered: [finance, flights, sports, finance, ...]

**Tuning**:
- Raise to require stronger consensus
- Lower if your intents have fuzzy boundaries

### Finding the Right Operating Point

```bash
# Sweep different threshold values
for ts in 0.40 0.45 0.50 0.55; do
    echo "=== min_top_score=$ts ==="
    uv run python scripts/bench.py --min-top-score $ts --json-out results_$ts.json
done
```

Plot precision vs coverage to find the Pareto frontier.

---

## Merino Integration

### Recommended Approach: Middleware

Intent classification should be **middleware** that runs before providers:

```
Request: "aapl stock price"
        │
        ▼
┌───────────────────────┐
│  IntentMiddleware     │  ← Classify query, attach to request
│  intent = "finance"   │
│  confidence = 0.92    │
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  Provider Manager     │  ← Route based on intent
│  → Only call finance  │
│    provider           │
└───────────────────────┘
        │
        ▼
Response: Stock price suggestions
```

### Integration Steps

1. **Copy the `vss/` package** into Merino (e.g., `merino/middleware/intent/`)

2. **Copy artifacts** into the package or fetch from storage

3. **Create middleware** that:
   - Loads classifier once at startup
   - Classifies each query
   - Attaches result to request state

4. **Modify provider routing** to use intent:
   ```python
   if intent == "finance" and confidence > 0.8:
       providers_to_call = ["finance"]
   elif intent == "flights" and confidence > 0.8:
       providers_to_call = ["flightaware"]
   # etc.
   ```

### Example Middleware

```python
class IntentMiddleware:
    def __init__(self, app):
        self.app = app
        self.classifier = None  # Lazy load

    def get_classifier(self):
        if self.classifier is None:
            store = FaissIntentIndex.load(...)
            embedder = Embedder("all-MiniLM-L6-v2")
            self.classifier = IntentClassifier(store, embedder)
        return self.classifier

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            query = Request(scope).query_params.get("q", "")
            if query:
                result = self.get_classifier().classify(query)
                scope["intent"] = result
        await self.app(scope, receive, send)
```

---

## Key Design Decisions

### Why FAISS (not a vector database)?

- FAISS is a library, not a service — no network calls, no infrastructure
- For our scale (~100-1000 exemplars), it's overkill-fast
- Search latency is ~0.01ms; embedding is the bottleneck
- If we scale to millions, FAISS supports approximate indexes

### Why SentenceTransformers?

- Standard library for text embeddings in Python
- Pre-trained models work well out of the box
- Easy to swap models (e.g., multilingual)
- Can fine-tune on our data if needed

### Why L2 normalization + Inner Product?

- Cosine similarity = dot product when vectors are unit length
- FAISS IndexFlatIP (inner product) is faster than computing cosine
- Standard trick in production vector search systems

### Why "exemplars" over "centroids"?

- Exemplars: One vector per training example — richer, more nuanced
- Centroids: One vector per intent (the mean) — smaller, but loses nuance
- Centroids fail on "multi-modal" intents (e.g., finance = stocks AND crypto)
- We build both; exemplars is the default

### Why these default thresholds?

The defaults (0.45, 0.05, 0.60) are conservative starting points:
- Optimized for high precision (don't route wrong)
- Accept lower recall (some good queries won't route)
- Tune based on your precision/recall requirements

---

## FAQ

### Q: How much training data do I need?

Start with 50-200 examples per intent. More is better for coverage, but diminishing returns past ~500.

### Q: How do I add a new intent?

1. Add labeled examples to `data/train.jsonl`
2. Add test examples to `data/test.jsonl`
3. Rebuild the index: `uv run python scripts/build_index.py`
4. Run benchmarks to verify: `uv run python scripts/bench.py`

### Q: How do I handle typos?

The embedding model has some robustness to typos ("aapl" vs "appl"), but severe typos may cause abstains. Options:
- Add common misspellings to training data
- Apply spell correction before classification
- Lower thresholds (but watch precision!)

### Q: What about multilingual queries?

`all-MiniLM-L6-v2` is English-focused. For multilingual:
- Use `paraphrase-multilingual-MiniLM-L12-v2`
- Add training examples in target languages
- Benchmark quality in each language

### Q: Can I fine-tune the embedding model?

Yes! SentenceTransformers supports fine-tuning. But start with pre-trained models — they often work well enough without tuning.

### Q: How do I handle "none" (out-of-distribution) queries?

The abstain thresholds handle this. Don't add "none" exemplars to the index — "none" isn't a coherent cluster. Instead, tune thresholds to reject out-of-distribution queries.

### Q: What's the latency breakdown?

Typical for all-MiniLM-L6-v2 on CPU:
- Model loading: 2-5 seconds (one-time, at startup)
- Encoding: 5-10ms per query (the bottleneck)
- Search: 0.01-0.05ms per query (negligible)

---

## License

This project is part of Mozilla's Merino ecosystem. See [LICENSE](LICENSE) for details.