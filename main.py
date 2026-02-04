"""
main.py - Initial exploration/demo script for FAISS + SentenceTransformers.

=============================================================================
WHAT THIS FILE IS
=============================================================================

This is NOT part of the production system. It's the original "hello world"
exploration script I wrote when first learning FAISS and SentenceTransformers.

I'm keeping it in the repo because:
    1. It's a simple, self-contained example of how the libraries work
    2. It's useful for demos and quick sanity checks
    3. It documents my initial learning process

For the ACTUAL intent classification system, see:
    - vss/intent_faiss.py  (core classification logic)
    - scripts/build_index.py  (build the index)
    - scripts/bench.py  (benchmark quality and latency)
    - scripts/query.py  (classify individual queries)

=============================================================================
WHAT THIS SCRIPT DEMONSTRATES
=============================================================================

1. Loading a SentenceTransformer model
2. Encoding sentences into embeddings
3. Computing pairwise similarity between embeddings
4. Building a FAISS index
5. Searching the index with query vectors

This is the "from scratch" version before I factored everything into
proper modules with abstraction and error handling.

=============================================================================
CONCEPTS ILLUSTRATED
=============================================================================

SENTENCE TRANSFORMERS:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)

    - Takes text → returns vectors (384 dimensions for MiniLM)
    - Similar meanings → similar vectors
    - The magic is in the pre-trained model weights

FAISS INDEX:
    idx = faiss.IndexFlatL2(embedding_dim)  # L2 = Euclidean distance
    idx.add(embeddings)                      # Add vectors to index
    D, I = idx.search(query_vectors, k)      # Find k nearest neighbors

    - D = distances (lower = more similar for L2)
    - I = indices (which vectors in the index matched)

NOTE: In the production code (intent_faiss.py), we use IndexFlatIP (inner product)
with normalized vectors instead of IndexFlatL2. This gives us cosine similarity,
which works better for text embeddings. This demo uses L2 for simplicity.

=============================================================================
USAGE
=============================================================================

    uv run python main.py

You'll see:
    - Embedding shape (6, 384) - 6 sentences, 384 dimensions each
    - Similarity matrix (6x6) - pairwise cosine similarities
    - Search results - which sentences are closest to each query

=============================================================================
"""

import faiss
from sentence_transformers import SentenceTransformer


def main():
    print("=" * 60)
    print("FAISS + SentenceTransformers Demo")
    print("=" * 60)
    print()

    # -------------------------------------------------------------------------
    # Step 1: Load the embedding model
    # -------------------------------------------------------------------------
    # This downloads/loads a pre-trained transformer model (~90MB).
    # "all-MiniLM-L6-v2" is a good balance of speed and quality.
    #
    # First run will download the model; subsequent runs use the cache.

    print("Loading SentenceTransformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("  Model loaded: all-MiniLM-L6-v2")
    print()

    # -------------------------------------------------------------------------
    # Step 2: Define some example sentences
    # -------------------------------------------------------------------------
    # These represent a mini "corpus" we want to search over.
    # Notice we have three themes: weather, flights, and airports.

    sentences = [
        "The weather is snowy today.",
        "It's so sunny outside!",
        "He drove to the stadium on a rainy day.",
        "AC 251 flight status.",
        "CA 14 flight delayed.",
        "Toronto Pearson airport is currently closed.",
    ]

    print("Corpus sentences:")
    for i, s in enumerate(sentences):
        print(f"  [{i}] {s}")
    print()

    # -------------------------------------------------------------------------
    # Step 3: Encode sentences into embeddings
    # -------------------------------------------------------------------------
    # This is the expensive step: each sentence goes through the transformer.
    # Output shape: (n_sentences, embedding_dim) = (6, 384)

    print("Encoding sentences into embeddings...")
    embeddings = model.encode(sentences)
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Each sentence → {embeddings.shape[1]}-dimensional vector")
    print()

    # -------------------------------------------------------------------------
    # Step 4: Compute pairwise similarities
    # -------------------------------------------------------------------------
    # SentenceTransformer has a built-in similarity function.
    # This computes cosine similarity between all pairs of sentences.
    #
    # The resulting matrix is symmetric: similarity[i][j] = similarity[j][i]
    # Diagonal is 1.0 (each sentence is identical to itself).

    print("Pairwise cosine similarities:")
    similarities = model.similarity(embeddings, embeddings)
    print(similarities)
    print()
    print("Note: Weather sentences (0,1,2) should be more similar to each other")
    print("than to flight sentences (3,4,5). Check the matrix!")
    print()

    # -------------------------------------------------------------------------
    # Step 5: Build a FAISS index
    # -------------------------------------------------------------------------
    # FAISS is a library for efficient similarity search.
    #
    # IndexFlatL2(d) creates an index for d-dimensional vectors using
    # L2 (Euclidean) distance. "Flat" means exact search (no approximation).
    #
    # For normalized vectors, L2 distance is related to cosine similarity:
    #   L2(a, b)^2 = 2 - 2*cos(a, b)  (when ||a|| = ||b|| = 1)
    #
    # So lower L2 distance = higher cosine similarity.

    print("Building FAISS index...")
    embedding_dim = embeddings.shape[1]
    idx = faiss.IndexFlatL2(embedding_dim)
    idx.add(embeddings)  # type: ignore[call-arg] # Add all sentence embeddings to the index
    print("  Index type: IndexFlatL2 (exact L2 distance search)")
    print(f"  Vectors in index: {idx.ntotal}")
    print()

    # -------------------------------------------------------------------------
    # Step 6: Define some queries
    # -------------------------------------------------------------------------
    # These are NEW sentences that we want to match against our corpus.
    # Notice they use different words but similar meanings.

    queries = [
        "Toronto weather forecast.",  # Should match weather sentences
        "Flight status AC 24.",  # Should match flight sentences
        "Best way to go to Pearson airport.",  # Should match airport sentence
        "grocery shopping ideas.",  # Out of distribution - shouldn't match well
    ]

    print("Query sentences:")
    for i, q in enumerate(queries):
        print(f"  [{i}] {q}")
    print()

    # -------------------------------------------------------------------------
    # Step 7: Encode queries and search
    # -------------------------------------------------------------------------
    # Same encoding process as the corpus. IMPORTANT: Must use the same model!

    print("Encoding queries and searching...")
    query_embeddings = model.encode(queries)

    # Search for k=2 nearest neighbors for each query
    k = 2
    D, I = idx.search(query_embeddings, k)  # type: ignore[call-arg] # noqa: E741
    # D = distances, shape (n_queries, k)
    # I = indices, shape (n_queries, k)

    print()
    print("=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)
    print()

    for query_idx, query in enumerate(queries):
        print(f'Query: "{query}"')
        print(f"  Nearest neighbors (k={k}):")
        for rank in range(k):
            corpus_idx = I[query_idx][rank]
            distance = D[query_idx][rank]
            # Lower distance = more similar (L2 distance)
            print(f"    [{rank + 1}] idx={corpus_idx}, distance={distance:.4f}")
            print(f'        "{sentences[corpus_idx]}"')
        print()

    # -------------------------------------------------------------------------
    # Observations
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("OBSERVATIONS")
    print("=" * 60)
    print()
    print("1. 'Toronto weather forecast' should match weather sentences (0, 1, or 2)")
    print("2. 'Flight status AC 24' should match flight sentences (3 or 4)")
    print("3. 'Best way to go to Pearson airport' should match sentence 5 (airport)")
    print(
        "4. 'grocery shopping ideas' is out-of-distribution - distances will be higher"
    )
    print()
    print("This demonstrates semantic search: matching by MEANING, not keywords.")
    print("The query 'Toronto weather forecast' matches 'The weather is snowy today'")
    print("even though they share only one word ('weather')!")
    print()
    print("-" * 60)
    print("For the full intent classification system, see:")
    print("  - scripts/build_index.py  (build production index)")
    print("  - scripts/bench.py        (run quality benchmarks)")
    print("  - scripts/query.py        (classify single queries)")
    print("-" * 60)


if __name__ == "__main__":
    main()
