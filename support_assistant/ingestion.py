from pathlib import Path
import re

import chromadb
from sentence_transformers import SentenceTransformer


# Paths
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"
MODEL_NAME = "all-MiniLM-L6-v2"


def chunk_text(text: str, max_chars: int = 400) -> list[str]:
    """Split policy text into small sentence-based chunks."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def main():
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Recreate collection so the script is safe to run again
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
    )

    all_chunks = []
    all_ids = []
    all_metadatas = []

    print("Reading policy documents...")

    for doc_path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = doc_path.read_text(encoding="utf-8").strip()

        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{doc_path.stem}_chunk_{index}")
            all_metadatas.append(
                {
                    "source": doc_path.name,
                    "chunk_index": index,
                }
            )

    print(f"Documents found: {len(list(DOCS_DIR.glob('doc_*.txt')))}")
    print(f"Chunks created: {len(all_chunks)}")

    print("Creating embeddings...")
    embeddings = model.encode(
        all_chunks,
        normalize_embeddings=True,
    ).tolist()

    print("Storing embeddings in ChromaDB...")

    collection.add(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas,
    )

    print("Ingestion completed successfully.")
    print(f"Stored chunks: {collection.count()}")


if __name__ == "__main__":
    main()