"""
bench.py - Benchmarking utilities for intent classification quality and latency.

=============================================================================
OVERVIEW
=============================================================================

This module provides tools to evaluate the intent classifier. There are two
types of benchmarks we care about:

1. QUALITY METRICS: Is the classifier making correct predictions?
   - Precision: When we predict "finance", how often are we right?
   - Recall: Of all actual "finance" queries, how many did we catch?
   - Coverage: What fraction of queries do we actually route (non-abstain)?
   - False positive rate: How often do we incorrectly route "none" queries?

2. LATENCY METRICS: How fast is the classifier?
   - Encoding latency (p50, p95, p99)
   - Search latency (p50, p95, p99)
   - Total latency (p50, p95, p99)

=============================================================================
WHY THESE SPECIFIC METRICS?
=============================================================================

For Merino's intent routing, our priorities are:

1. HIGH PRECISION (most important)
   - When we route a query to a specialized provider, we must be confident.
   - False positives = bad user experience. Routing "python list sort" to
     flights is worse than not routing a legitimate flights query.

2. ACCEPTABLE RECALL (secondary)
   - We want to capture as many high-intent queries as we can.
   - But we trade off recall for precision. It's OK to miss some queries.

3. CONTROLLED COVERAGE
   - Coverage tells us what fraction of queries we're routing.
   - Too high coverage with low precision = we're being too aggressive.
   - Too low coverage = we might be too conservative (leaving money on table).

4. LOW LATENCY
   - Merino is in the critical path of address bar suggestions.
   - p50 should be <20ms, p99 should be <50ms (example targets).
   - Percentiles matter more than mean: users experience the tail.

=============================================================================
HOW TO USE THIS MODULE
=============================================================================

Typical benchmark workflow:

    1. Build the index (using build_index.py)
    2. Run benchmark against test.jsonl (using bench.py script)
    3. Look at:
       - provider_fp_rate: Should be very low (<<5%)
       - Per-intent precision: Should be high (>90%)
       - Per-intent recall: Trade off against precision
       - Latency p95: Should meet your SLA

    4. If results are bad:
       - Low precision? Raise thresholds (min_top_score, min_confidence)
       - Low recall? Lower thresholds (but watch precision!)
       - High latency? Profile encode step, consider smaller model

=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from vss.intent_faiss import TimingsMs

# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass(frozen=True)
class Metrics:
    """
    Container for all quality metrics from a benchmark run.

    Attributes:
        per_intent: Dict mapping intent -> metrics dict
                    Each intent has: tp, fp, fn, precision, recall, f1
        overall: Dict with aggregate metrics:
                 n, accuracy, coverage, abstain_rate, provider_fp_rate, etc.
    """

    per_intent: Dict[str, Dict[str, float]]
    overall: Dict[str, float]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def percentile_ms(values: List[float], p: float) -> float:
    """
    Compute a percentile from a list of values.

    We report percentiles (p50, p95, p99) rather than mean for latency because:
    - Mean hides tail latency (one slow request affects user experience)
    - p50 = median, the "typical" experience
    - p95/p99 = the "bad day" experience, what slow users see

    Args:
        values: List of latency values in milliseconds
        p: Percentile to compute (0-100)

    Returns:
        The p-th percentile value
    """
    if not values:
        return 0.0
    arr = np.asarray(values, dtype="float64")
    return float(np.percentile(arr, p))


# =============================================================================
# QUALITY METRICS
# =============================================================================


def compute_metrics(
    truth: List[str],
    pred: List[str],
    positive_intents: List[str],
) -> Metrics:
    """
    Compute classification quality metrics.

    This function computes standard classification metrics, with special
    attention to the "open-set" nature of our problem (where "none" is a
    valid prediction meaning "abstain").

    TERMINOLOGY:
    ------------
    - truth: The ground-truth labels from test.jsonl
    - pred: The predictions from the classifier
    - positive_intents: The "real" intents we want to route to (finance, flights, sports)
                        Excludes "none", which means "abstain" or "unknown"

    PER-INTENT METRICS:
    -------------------
    For each provider intent (finance, flights, sports), we compute:

    - TP (True Positive): We predicted X, and it was actually X
    - FP (False Positive): We predicted X, but it was actually something else
    - FN (False Negative): It was actually X, but we predicted something else

    - Precision = TP / (TP + FP)
      "When we say 'finance', how often are we right?"

    - Recall = TP / (TP + FN)
      "Of all actual finance queries, what fraction did we catch?"

    - F1 = 2 * (Precision * Recall) / (Precision + Recall)
      Harmonic mean of precision and recall.

    OVERALL METRICS:
    ----------------
    - accuracy: (correct predictions) / (total predictions)
      Note: This can be misleading if classes are imbalanced.

    - coverage: (non-abstain predictions) / (total predictions)
      What fraction of queries do we actually route?

    - abstain_rate: 1 - coverage
      What fraction of queries do we decline to route?

    - provider_fp_rate: (queries labeled "none" but routed) / (total "none" queries)
      THIS IS THE KEY SAFETY METRIC. If this is high, we're routing queries
      that shouldn't be routed, which means bad user experience.

    Args:
        truth: List of ground-truth intent labels
        pred: List of predicted intent labels (including "none" for abstain)
        positive_intents: List of provider intents (excluding "none")

    Returns:
        A Metrics object with per_intent and overall dictionaries
    """
    if len(truth) != len(pred):
        raise ValueError("truth and pred length mismatch")

    # Sort intents for consistent ordering in reports
    intents = sorted(set(positive_intents))

    # -------------------------------------------------------------------------
    # Per-intent metrics (standard precision/recall/F1)
    # -------------------------------------------------------------------------

    per: Dict[str, Dict[str, float]] = {}

    for intent in intents:
        tp = 0  # True positives
        fp = 0  # False positives
        fn = 0  # False negatives

        for t, p in zip(truth, pred):
            if p == intent and t == intent:
                # Predicted intent, was actually intent → TP
                tp += 1
            elif p == intent and t != intent:
                # Predicted intent, was actually something else → FP
                fp += 1
            elif p != intent and t == intent:
                # Predicted something else, was actually intent → FN
                fn += 1
            # Note: TN (true negative) = predicted not-X and was not-X
            # We don't need TN for precision/recall, but it exists.

        # Compute precision, recall, F1 with divide-by-zero protection
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per[intent] = {
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    # -------------------------------------------------------------------------
    # Overall metrics
    # -------------------------------------------------------------------------

    n = len(truth)

    # Simple accuracy: what fraction of predictions exactly matched truth?
    correct = sum(1 for t, p in zip(truth, pred) if t == p)
    accuracy = correct / n if n > 0 else 0.0

    # Provider false positive rate: of all "none" queries, how many did we route?
    # This is the KEY SAFETY METRIC. We want this to be VERY LOW.
    #
    # Example:
    #   truth = ["finance", "none", "none", "flights", "none"]
    #   pred  = ["finance", "finance", "none", "flights", "none"]
    #
    #   none_total = 3 (queries 1, 2, 4)
    #   provider_fp = 1 (query 1 was "none" but we predicted "finance")
    #   provider_fp_rate = 1/3 = 0.33 (33% of none queries were incorrectly routed)
    #
    none_total = sum(1 for t in truth if t == "none")
    provider_fp = sum(1 for t, p in zip(truth, pred) if t == "none" and p != "none")
    provider_fp_rate = provider_fp / none_total if none_total > 0 else 0.0

    # Coverage: what fraction of queries did we route (not abstain)?
    # Higher coverage means we're routing more queries.
    # But if precision is low, high coverage is BAD.
    coverage = sum(1 for p in pred if p != "none") / n if n > 0 else 0.0
    abstain_rate = 1.0 - coverage

    overall = {
        "n": float(n),
        "accuracy": float(accuracy),
        "coverage": float(coverage),
        "abstain_rate": float(abstain_rate),
        "provider_fp_rate": float(provider_fp_rate),
        "none_total": float(none_total),
        "provider_fp": float(provider_fp),
    }

    return Metrics(per_intent=per, overall=overall)


# =============================================================================
# LATENCY METRICS
# =============================================================================


def summarize_latency(timings: List[TimingsMs]) -> Dict[str, float]:
    """
    Compute latency percentiles from a list of timing measurements.

    WHY PERCENTILES?
    ----------------
    Mean latency is misleading. If 99% of requests take 5ms and 1% take 500ms,
    the mean is ~10ms, but users see that 500ms tail!

    We report:
    - p50 (median): The "typical" user experience
    - p95: What 5% of users experience (the "slow" cases)
    - p99: What 1% of users experience (the "really slow" cases)

    For SLA purposes, you typically care about p95 or p99. Example SLAs:
    - "p50 < 10ms" (typical case is fast)
    - "p99 < 50ms" (even worst case is acceptable)

    WHY BREAK DOWN BY encode/search/total?
    --------------------------------------
    - encode_ms: Time spent in the transformer model (embedding the query)
    - search_ms: Time spent searching the FAISS index
    - total_ms: End-to-end time including all overhead

    In practice, encode dominates (5-10ms), search is negligible (0.01ms).
    If total >> encode + search, there's unexpected overhead to investigate.

    Args:
        timings: List of TimingsMs from classify_with_timings() calls

    Returns:
        Dict with keys like "encode_p50_ms", "encode_p95_ms", "encode_p99_ms", etc.
    """
    # Extract each timing component into separate lists
    encode = [t.encode_ms for t in timings]
    search = [t.search_ms for t in timings]
    total = [t.total_ms for t in timings]

    return {
        # Encoding latency (the bottleneck)
        "encode_p50_ms": percentile_ms(encode, 50),
        "encode_p95_ms": percentile_ms(encode, 95),
        "encode_p99_ms": percentile_ms(encode, 99),
        # Search latency (usually negligible)
        "search_p50_ms": percentile_ms(search, 50),
        "search_p95_ms": percentile_ms(search, 95),
        "search_p99_ms": percentile_ms(search, 99),
        # Total latency (what the caller experiences)
        "total_p50_ms": percentile_ms(total, 50),
        "total_p95_ms": percentile_ms(total, 95),
        "total_p99_ms": percentile_ms(total, 99),
    }
