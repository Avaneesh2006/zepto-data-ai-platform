from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "zepto_policies"


def main():
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_collection(COLLECTION_NAME)

    query = "How long do I have to report a damaged grocery item?"

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
    )

    print("\nQUERY:")
    print(query)

    print("\nTOP 3 RESULTS:")

    for i, document in enumerate(results["documents"][0], start=1):
        metadata = results["metadatas"][0][i - 1]
        distance = results["distances"][0][i - 1]

        print(f"\n{i}. Source: {metadata['source']}")
        print(f"   Distance: {distance:.4f}")
        print(f"   Text: {document}")


if __name__ == "__main__":
    main()