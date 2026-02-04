"""
query.py - Interactive script to classify a single query.

=============================================================================
WHAT THIS SCRIPT DOES
=============================================================================

This is a debugging and exploration tool. It classifies a single query and
shows you exactly what the classifier is doing:

    - The predicted intent (or "none" if abstained)
    - The confidence score
    - Whether it abstained and why (which threshold failed)
    - The top-k neighbors that influenced the decision

This is NOT for production use. It's for:
    - Debugging: "Why did this query get classified as X?"
    - Exploration: "What happens if I try this query?"
    - Demos: Showing stakeholders how the system works
    - Development: Testing changes before running full benchmarks

=============================================================================
UNDERSTANDING THE OUTPUT
=============================================================================

Example output:

    Classification
    query: plane tickets to nyc
    predicted_intent: flights
    confidence: 0.823456
    abstained: False
    top_score: 0.891234
    margin: 0.156789

    Top neighbors
    id=6 score=0.891234 intent=flights text=flights from sfo to jfk
    id=7 score=0.734445 intent=flights text=lax to sea flights
    id=4 score=0.612233 intent=flights text=delta 215 flight status
    ...

How to read this:

    predicted_intent: What the classifier thinks the query is about.
                      "none" means it abstained (not confident enough).

    confidence: What fraction of the neighborhood "mass" belongs to the
                predicted intent. 0.82 means 82% of the weighted neighbor
                votes went to "flights".

    abstained: Did we abstain? If True, predicted_intent will be "none".
               Check top_score, margin, and confidence to see which
               threshold caused the abstain.

    top_score: Similarity of the closest neighbor. If this is low (<0.45
               by default), the query is "out of distribution" - unlike
               anything in the training set.

    margin: Gap between best and second-best neighbor scores. If this is
            small (<0.05 by default), the query is ambiguous - multiple
            intents are plausible.

    Top neighbors: The actual training examples that influenced the
                   classification. This is great for debugging because
                   you can see exactly WHY the classifier made its decision.

=============================================================================
USAGE
=============================================================================

    # Basic usage
    uv run python scripts/query.py "aapl stock price"

    # Try different queries
    uv run python scripts/query.py "flights to paris"
    uv run python scripts/query.py "lakers game score"
    uv run python scripts/query.py "how to boil eggs"  # Should abstain!

    # Use centroids index instead of exemplars
    uv run python scripts/query.py "aapl stock" --index-kind centroids

    # Experiment with thresholds
    uv run python scripts/query.py "aapl stock" --min-top-score 0.60

    # Use more neighbors
    uv run python scripts/query.py "aapl stock" --k 16

=============================================================================
COMMON DEBUGGING SCENARIOS
=============================================================================

Scenario 1: Query classified incorrectly
-----------------------------------------
    $ uv run python scripts/query.py "airport parking"
    predicted_intent: flights  # Oops, should probably be "none"

    Look at the neighbors:
    - Are there training examples that are misleadingly similar?
    - Is the confidence high or low?
    - Consider: add more diverse training examples, or raise thresholds

Scenario 2: Query abstained when it shouldn't
---------------------------------------------
    $ uv run python scripts/query.py "stock market news"
    predicted_intent: none  # Should be "finance"!

    Look at which threshold failed:
    - top_score too low? Training examples don't cover this phrasing.
    - margin too low? Ambiguous between multiple intents.
    - confidence too low? Neighbors are scattered across intents.

    Consider: add training examples, or lower thresholds (carefully!)

Scenario 3: Understanding a correct classification
--------------------------------------------------
    $ uv run python scripts/query.py "msft earnings"
    predicted_intent: finance  # Correct!

    The neighbors show WHY it worked:
    - "nvda earnings" is in the training set with score 0.92
    - Multiple finance examples in the neighborhood

=============================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vss.intent_faiss import Embedder, FaissIntentIndex, IntentClassifier


def main() -> None:
    """
    Main entry point for the query script.

    This function:
        1. Parses command-line arguments (including the query)
        2. Loads the pre-built FAISS index and embedding model
        3. Classifies the query
        4. Prints detailed results for debugging
    """

    # -------------------------------------------------------------------------
    # Argument parsing
    # -------------------------------------------------------------------------

    p = argparse.ArgumentParser(
        description="Classify a single query and show detailed results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # The query is a positional argument (required)
    p.add_argument(
        "query",
        help="The query to classify (e.g., 'aapl stock price')",
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
            "Which index to use: 'exemplars' (one vector per training example) "
            "or 'centroids' (one vector per intent)"
        ),
    )

    p.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name (must match the index!)",
    )

    # -------------------------------------------------------------------------
    # Classification hyperparameters
    # -------------------------------------------------------------------------

    p.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of neighbors to retrieve",
    )

    p.add_argument(
        "--min-top-score",
        type=float,
        default=0.45,
        help="Minimum similarity for best match (abstain if below)",
    )

    p.add_argument(
        "--min-margin",
        type=float,
        default=0.05,
        help="Minimum gap between best and second-best (abstain if below)",
    )

    p.add_argument(
        "--min-confidence",
        type=float,
        default=0.60,
        help="Minimum confidence for winning intent (abstain if below)",
    )

    args = p.parse_args()

    # -------------------------------------------------------------------------
    # Load index artifacts
    # -------------------------------------------------------------------------

    artifacts = Path(args.artifacts_dir)

    if args.index_kind == "exemplars":
        index_path = artifacts / "exemplars.faiss"
        meta_path = artifacts / "exemplars_meta.json"
    else:
        index_path = artifacts / "centroids.faiss"
        meta_path = artifacts / "centroids_meta.json"

    # Check that artifacts exist (common mistake: forgetting to build first)
    if not index_path.exists():
        print(f"ERROR: Index file not found: {index_path}")
        print("")
        print("Did you forget to build the index? Run:")
        print("  uv run python scripts/build_index.py")
        return

    store = FaissIntentIndex.load(str(index_path), str(meta_path))

    # Warn if model mismatch (results will be garbage)
    if store.model_name != args.model:
        print("=" * 60)
        print("WARNING: MODEL MISMATCH!")
        print("=" * 60)
        print(f"Index was built with:  {store.model_name}")
        print(f"You specified:         {args.model}")
        print("Results may be invalid!")
        print("=" * 60)
        print("")

    # -------------------------------------------------------------------------
    # Initialize classifier
    # -------------------------------------------------------------------------

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
    # Classify the query
    # -------------------------------------------------------------------------

    cls = clf.classify(args.query)

    # -------------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("CLASSIFICATION RESULT")
    print("=" * 60)
    print(f"query:            {args.query}")
    print(f"predicted_intent: {cls.predicted_intent}")
    print(f"confidence:       {cls.confidence:.6f}")
    print(f"abstained:        {cls.abstained}")

    print("")
    print("-" * 60)
    print("Decision signals (used for abstain logic):")
    print("-" * 60)
    print(f"top_score:        {cls.top_score:.6f}  (threshold: {args.min_top_score})")
    print(f"margin:           {cls.margin:.6f}  (threshold: {args.min_margin})")
    print(f"confidence:       {cls.confidence:.6f}  (threshold: {args.min_confidence})")

    # Explain WHY we abstained (if we did)
    if cls.abstained:
        print("")
        print(">>> ABSTAINED because:")
        if cls.top_score < args.min_top_score:
            print(
                f"    - top_score ({cls.top_score:.4f}) < min_top_score ({args.min_top_score})"
            )
            print("      (Query is unlike anything in training set)")
        if cls.margin < args.min_margin:
            print(f"    - margin ({cls.margin:.4f}) < min_margin ({args.min_margin})")
            print("      (Top matches are too close; ambiguous query)")
        if cls.confidence < args.min_confidence:
            print(
                f"    - confidence ({cls.confidence:.4f}) < min_confidence ({args.min_confidence})"
            )
            print("      (Neighborhood is scattered across multiple intents)")

    print("")
    print("=" * 60)
    print(f"TOP {len(cls.neighbors)} NEIGHBORS")
    print("=" * 60)
    print("")
    print("These are the training examples most similar to your query.")
    print("The classifier aggregates their intent labels (weighted by score)")
    print("to make its prediction.")
    print("")

    for i, n in enumerate(cls.neighbors):
        # Highlight neighbors that match the predicted intent
        marker = "→" if n.intent == cls.predicted_intent else " "
        print(
            f'  {marker} [{i + 1}] score={n.score:.4f}  intent={n.intent:<10}  text="{n.text}"'
        )

    # -------------------------------------------------------------------------
    # Summary of neighbor aggregation
    # -------------------------------------------------------------------------

    print("")
    print("-" * 60)
    print("Intent score aggregation:")
    print("-" * 60)

    # Recompute the aggregation for display
    by_intent: dict[str, float] = {}
    total_mass = 0.0
    for n in cls.neighbors:
        mass = max(0.0, n.score)
        by_intent[n.intent] = by_intent.get(n.intent, 0.0) + mass
        total_mass += mass

    # Sort by score descending
    sorted_intents = sorted(by_intent.items(), key=lambda x: x[1], reverse=True)

    for intent, score in sorted_intents:
        pct = (score / total_mass * 100) if total_mass > 0 else 0
        marker = "→" if intent == cls.predicted_intent else " "
        print(f"  {marker} {intent:<10}  total_score={score:.4f}  ({pct:.1f}%)")

    print("")


if __name__ == "__main__":
    main()
