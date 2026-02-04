"""
bench.py - Benchmark script for evaluating intent classifier quality and latency.

=============================================================================
WHAT THIS SCRIPT DOES
=============================================================================

This script evaluates the intent classifier against a labeled test set. It
measures both quality (accuracy, precision, recall) and performance (latency).

The workflow is:

    1. Load a pre-built FAISS index (from build_index.py)
    2. Load test examples (queries with known intent labels)
    3. Classify each test query
    4. Compare predictions to ground truth
    5. Compute quality metrics (precision, recall, false positive rate)
    6. Compute latency percentiles (p50, p95, p99)
    7. Print a report (and optionally save to JSON)

=============================================================================
WHY BENCHMARK?
=============================================================================

Before deploying to production, you MUST understand:

1. QUALITY: Is the classifier accurate enough?
   - High precision on provider intents (finance, flights, sports)
   - Low false positive rate on "none" queries (don't route random queries!)
   - Acceptable recall (catching enough high-intent queries)

2. PERFORMANCE: Is the classifier fast enough?
   - Merino is in the address bar critical path
   - Latency budgets are tight (example: p99 < 50ms)
   - Encoding is the bottleneck, search is negligible

3. THRESHOLD TUNING: Are the abstain thresholds well-calibrated?
   - min_top_score, min_margin, min_confidence control precision/recall tradeoff
   - Run benchmarks with different thresholds to find the right operating point

=============================================================================
INTERPRETING RESULTS
=============================================================================

Key metrics to watch:

    provider_fp_rate:
        - This is "of queries that should NOT be routed, how many did we route?"
        - Should be VERY LOW (<5%, ideally <1%)
        - If this is high, RAISE your thresholds

    per-intent precision:
        - "When we predict 'finance', how often is it actually finance?"
        - Should be HIGH (>90%, ideally >95%)
        - Low precision = bad user experience

    per-intent recall:
        - "Of all actual finance queries, what fraction did we catch?"
        - Trade off against precision
        - Lower recall is acceptable if precision is high

    coverage / abstain_rate:
        - What fraction of queries do we route vs abstain?
        - High abstain rate (>50%) is OK if precision is high
        - We're being conservative: false negatives > false positives

    latency p95/p99:
        - The "slow case" user experience
        - Should meet your SLA requirements

=============================================================================
USAGE
=============================================================================

    # Basic usage (uses defaults)
    uv run python scripts/bench.py

    # Full usage with all options
    uv run python scripts/bench.py \
        --test data/test.jsonl \
        --artifacts-dir artifacts \
        --index-kind exemplars \
        --model all-MiniLM-L6-v2 \
        --k 8 \
        --min-top-score 0.45 \
        --min-margin 0.05 \
        --min-confidence 0.60 \
        --json-out results.json

    # Compare exemplars vs centroids
    uv run python scripts/bench.py --index-kind exemplars
    uv run python scripts/bench.py --index-kind centroids

    # Experiment with different thresholds
    uv run python scripts/bench.py --min-top-score 0.50 --min-confidence 0.70

=============================================================================
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from vss.bench import compute_metrics, summarize_latency
from vss.intent_faiss import (
    Embedder,
    FaissIntentIndex,
    IntentClassifier,
    read_jsonl_examples,
)


def main() -> None:
    """
    Main entry point for the benchmarking script.

    This function:
        1. Parses command-line arguments
        2. Loads the pre-built FAISS index and embedding model
        3. Runs classification on all test examples
        4. Computes quality and latency metrics
        5. Prints a human-readable report
        6. Optionally saves results to JSON
    """

    # -------------------------------------------------------------------------
    # Argument parsing
    # -------------------------------------------------------------------------

    p = argparse.ArgumentParser(
        description="Benchmark intent classifier quality and latency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--test",
        default="data/test.jsonl",
        help="Path to test data (JSONL format with 'text' and 'intent' fields)",
    )

    p.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory containing pre-built index artifacts",
    )

    p.add_argument(
        "--index-kind",
        choices=["exemplars", "centroids"],
        default="exemplars",
        help=(
            "Which index to benchmark: 'exemplars' (one vector per training example) "
            "or 'centroids' (one vector per intent). Exemplars is usually better."
        ),
    )

    p.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help=(
            "SentenceTransformer model name. MUST match the model used to build "
            "the index! If mismatched, results will be garbage."
        ),
    )

    # -------------------------------------------------------------------------
    # Classification hyperparameters
    # -------------------------------------------------------------------------
    # These control the precision/recall tradeoff. Higher values = more
    # conservative (more abstaining, higher precision, lower recall).

    p.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of neighbors to retrieve from FAISS",
    )

    p.add_argument(
        "--min-top-score",
        type=float,
        default=0.45,
        help=(
            "Minimum similarity score for the best match. If the closest neighbor "
            "has score below this, we abstain. Range: 0-1 for normalized vectors."
        ),
    )

    p.add_argument(
        "--min-margin",
        type=float,
        default=0.05,
        help=(
            "Minimum gap between best and second-best scores. If the top two "
            "matches are too close, we abstain (ambiguous query)."
        ),
    )

    p.add_argument(
        "--min-confidence",
        type=float,
        default=0.60,
        help=(
            "Minimum confidence (fraction of neighborhood belonging to winning "
            "intent). If no intent dominates the top-k, we abstain."
        ),
    )

    p.add_argument(
        "--json-out",
        default="",
        help="If provided, write detailed results to this JSON file",
    )

    args = p.parse_args()

    # -------------------------------------------------------------------------
    # Load index artifacts
    # -------------------------------------------------------------------------
    # We load the pre-built FAISS index and its metadata. The metadata tells us
    # which model was used to build the index (important for sanity checking).

    artifacts = Path(args.artifacts_dir)

    if args.index_kind == "exemplars":
        index_path = artifacts / "exemplars.faiss"
        meta_path = artifacts / "exemplars_meta.json"
    else:
        index_path = artifacts / "centroids.faiss"
        meta_path = artifacts / "centroids_meta.json"

    store = FaissIntentIndex.load(str(index_path), str(meta_path))

    # -------------------------------------------------------------------------
    # Model mismatch warning
    # -------------------------------------------------------------------------
    # This is a common gotcha: if you build with model A but query with model B,
    # the embeddings are incompatible and search results will be nonsense.
    # We warn loudly if this happens.

    if store.model_name != args.model:
        print("=" * 60)
        print("WARNING: MODEL MISMATCH!")
        print("=" * 60)
        print(f"Index was built with:  {store.model_name}")
        print(f"You specified:         {args.model}")
        print("")
        print("This will produce INVALID results. Either:")
        print("  1. Rebuild the index with --model", args.model)
        print("  2. Run this script with --model", store.model_name)
        print("=" * 60)
        print("")

    # -------------------------------------------------------------------------
    # Initialize classifier
    # -------------------------------------------------------------------------
    # The classifier combines:
    #   - The FAISS index (for nearest neighbor search)
    #   - The embedder (for encoding queries)
    #   - The decision thresholds (for abstain logic)

    embedder = Embedder(args.model)

    clf = IntentClassifier(
        index=store,
        embedder=embedder,
        k=int(args.k),
        min_top_score=float(args.min_top_score),
        min_margin=float(args.min_margin),
        min_confidence=float(args.min_confidence),
    )

    # -------------------------------------------------------------------------
    # Load test data
    # -------------------------------------------------------------------------
    # Test data format is the same as training data: JSONL with text and intent.
    # The intent field is the ground truth we compare against.
    #
    # IMPORTANT: Test data should include:
    #   - Positive examples for each provider intent (finance, flights, sports)
    #   - Negative examples labeled "none" (queries that should NOT route)
    #
    # The "none" examples are critical for measuring false positive rate!

    tests = read_jsonl_examples(args.test)

    # -------------------------------------------------------------------------
    # Warmup
    # -------------------------------------------------------------------------
    # The first few inference calls are often slower due to:
    #   - JIT compilation in PyTorch
    #   - CPU cache warming
    #   - Thread pool initialization
    #
    # We run a few warmup queries so our timing measurements are realistic.
    # In production, this happens naturally as the service handles initial requests.

    warmup_texts = [t.text for t in tests[:10]]
    for w in warmup_texts:
        clf.classify(w)  # Discard results, just warming up

    # -------------------------------------------------------------------------
    # Run classification on all test examples
    # -------------------------------------------------------------------------
    # We collect:
    #   - truth: Ground truth labels from test.jsonl
    #   - pred: Predicted labels from the classifier
    #   - timings: Timing breakdown for each classification

    truth: List[str] = []
    pred: List[str] = []
    timings = []

    for ex in tests:
        cls, t = clf.classify_with_timings(ex.text)
        truth.append(ex.intent)
        pred.append(cls.predicted_intent)
        timings.append(t)

    # -------------------------------------------------------------------------
    # Compute metrics
    # -------------------------------------------------------------------------
    # Quality metrics compare predictions to ground truth.
    # We need to know which intents are "positive" (provider intents we route to)
    # vs "none" (which means abstain). We get positive intents from train.jsonl.

    positive_intents = sorted(
        {e.intent for e in read_jsonl_examples("data/train.jsonl")}
    )
    metrics = compute_metrics(truth=truth, pred=pred, positive_intents=positive_intents)

    # Latency metrics summarize timing percentiles
    latency = summarize_latency(timings)

    # -------------------------------------------------------------------------
    # Print report
    # -------------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("BENCHMARK CONFIGURATION")
    print("=" * 60)
    print(f"Index kind:      {args.index_kind}")
    print(f"Model:           {args.model}")
    print(f"Test examples:   {len(tests)}")
    print(f"k:               {args.k}")
    print(f"min_top_score:   {args.min_top_score}")
    print(f"min_margin:      {args.min_margin}")
    print(f"min_confidence:  {args.min_confidence}")

    print("")
    print("=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)
    # Print overall metrics with explanatory notes
    print(f"n:               {metrics.overall['n']:.0f}  (total test examples)")
    print(f"accuracy:        {metrics.overall['accuracy']:.4f}  (fraction correct)")
    print(f"coverage:        {metrics.overall['coverage']:.4f}  (fraction routed)")
    print(
        f"abstain_rate:    {metrics.overall['abstain_rate']:.4f}  (fraction not routed)"
    )
    print("")
    print(">>> SAFETY METRIC (should be LOW):")
    print(f"provider_fp_rate: {metrics.overall['provider_fp_rate']:.4f}  ", end="")
    print(
        f"({int(metrics.overall['provider_fp'])}/{int(metrics.overall['none_total'])} ",
        end="",
    )
    print("'none' queries incorrectly routed)")

    print("")
    print("=" * 60)
    print("PER-INTENT METRICS")
    print("=" * 60)
    for intent, d in metrics.per_intent.items():
        print(f"\n  {intent.upper()}")
        print(
            f"    precision: {d['precision']:.4f}  (when we predict {intent}, how often correct?)"
        )
        print(
            f"    recall:    {d['recall']:.4f}  (of actual {intent}, how many caught?)"
        )
        print(f"    f1:        {d['f1']:.4f}  (harmonic mean)")
        print(f"    tp/fp/fn:  {int(d['tp'])}/{int(d['fp'])}/{int(d['fn'])}")

    print("")
    print("=" * 60)
    print("LATENCY METRICS (milliseconds)")
    print("=" * 60)
    print("")
    print("  Encoding (transformer forward pass - the bottleneck):")
    print(f"    p50:  {latency['encode_p50_ms']:7.3f} ms")
    print(f"    p95:  {latency['encode_p95_ms']:7.3f} ms")
    print(f"    p99:  {latency['encode_p99_ms']:7.3f} ms")
    print("")
    print("  Search (FAISS nearest neighbor - negligible):")
    print(f"    p50:  {latency['search_p50_ms']:7.3f} ms")
    print(f"    p95:  {latency['search_p95_ms']:7.3f} ms")
    print(f"    p99:  {latency['search_p99_ms']:7.3f} ms")
    print("")
    print("  Total (end-to-end):")
    print(f"    p50:  {latency['total_p50_ms']:7.3f} ms")
    print(f"    p95:  {latency['total_p95_ms']:7.3f} ms")
    print(f"    p99:  {latency['total_p99_ms']:7.3f} ms")

    # -------------------------------------------------------------------------
    # Optional JSON output
    # -------------------------------------------------------------------------
    # For programmatic analysis, CI/CD integration, or historical tracking,
    # you can save the full results to JSON.

    if args.json_out:
        out = {
            "config": {
                "index_kind": args.index_kind,
                "model": args.model,
                "k": args.k,
                "min_top_score": args.min_top_score,
                "min_margin": args.min_margin,
                "min_confidence": args.min_confidence,
            },
            "overall": metrics.overall,
            "per_intent": metrics.per_intent,
            "latency": latency,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print("")
        print(f"Wrote JSON report: {args.json_out}")

    print("")


if __name__ == "__main__":
    main()
