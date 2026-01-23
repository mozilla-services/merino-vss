import faiss

from sentence_transformers import SentenceTransformer


def main():
    print("Hello from merino-vss!")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    sentences = [
        "The weather is snowy today.",
        "It's so sunny outside!",
        "He drove to the stadium on a rainy day.",
        "AC 251 flight status.",
        "CA 14 flight delayed.",
        "Toronto Pearson airport is currently closed.",
    ]

    embeddings = model.encode(sentences)
    print(embeddings.shape)

    similarities = model.similarity(embeddings, embeddings)
    print(similarities)

    idx = faiss.IndexFlatL2(embeddings.shape[1])

    idx.add(embeddings)

    queries = [
        "Toronto weather forecast.",
        "Flight status AC 24.",
        "Best way to go to Pearson airport.",
        "grocery shopping ideas."
    ]

    query_embeddings = model.encode(queries)

    k = 2
    D, I = idx.search(query_embeddings, k)

    print("Indices of nearest neighbors:\n", I)
    print("Distance to nearest neighbors:\n", D)


if __name__ == "__main__":
    main()
