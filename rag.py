"""
RAG layer for the return policy document.

- Reads data/return_policy.md
- Chunks by markdown H2 sections (one chunk per policy section)
- Embeds with a local sentence-transformers model (no API cost)
- Stores in ChromaDB persistently at ./chroma_db/
- Exposes get_return_policy(query) as the tool the agent calls
"""
import re
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions


# Config
PROJECT_ROOT = Path(__file__).parent
POLICY_FILE = PROJECT_ROOT / "data" / "return_policy.md"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "return_policy"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 22MB, 384-dim, fast, good enough


def chunk_policy_by_sections(text: str) -> list[dict]:
    """
    Split policy markdown by H2 sections (## N. Section Name).
    Each section becomes one chunk that stays semantically coherent.
    """
    chunks = []
    # Split on lines starting with '## ' — keeps the delimiter with the following block
    parts = re.split(r'(?=^## )', text, flags=re.MULTILINE)

    for part in parts:
        part = part.strip()
        if not part.startswith('## '):
            continue  # skips the H1 title, preamble, etc.

        lines = part.split('\n', 1)
        header_line = lines[0].strip()          # e.g. "## 3. Perishable items"
        content = lines[1].strip() if len(lines) > 1 else ""

        # Extract number and title from header (e.g. "3", "Perishable items")
        header = header_line.replace('## ', '').strip()
        m = re.match(r'^(\d+)\.\s*(.+)', header)
        section_number = m.group(1) if m else ""
        section_title = m.group(2) if m else header

        # Include the header inside the chunk text so retrieval scores well
        # even when the user's query matches the section title
        chunk_text = f"{header_line}\n\n{content}"

        chunks.append({
            'section_number': section_number,
            'section_title': section_title,
            'content': chunk_text,
        })

    return chunks


def get_collection():
    """Get or create the persistent Chroma collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"description": "Return policy sections for customer support agent"},
    )

    return collection


def build_index(force_rebuild: bool = False) -> int:
    """
    Chunk the policy and load it into Chroma.
    Skips if the index is already built unless force_rebuild=True.
    """
    collection = get_collection()

    if collection.count() > 0 and not force_rebuild:
        print(f"Index already built ({collection.count()} chunks). "
              f"Pass force_rebuild=True to reindex.")
        return collection.count()

    # Clear existing chunks if rebuilding
    if collection.count() > 0:
        existing_ids = collection.get()['ids']
        collection.delete(ids=existing_ids)

    # Load policy and chunk
    if not POLICY_FILE.exists():
        raise FileNotFoundError(f"Policy file not found: {POLICY_FILE}")

    policy_text = POLICY_FILE.read_text()
    chunks = chunk_policy_by_sections(policy_text)

    if not chunks:
        raise ValueError(
            f"No sections found in {POLICY_FILE}. "
            "Make sure the policy has '## N. Section Name' headers."
        )

    # Insert into Chroma
    collection.add(
        ids=[f"section_{c['section_number']}" for c in chunks],
        documents=[c['content'] for c in chunks],
        metadatas=[{
            'section_number': c['section_number'],
            'section_title': c['section_title'],
        } for c in chunks],
    )

    print(f"Indexed {len(chunks)} policy sections")
    return len(chunks)


def get_return_policy(query: str, top_k: int = 3) -> list[dict]:
    """
    Query the policy index. This is the function your agent will call as a tool.

    Args:
        query: Natural language question about the return policy.
        top_k: How many sections to return (default 3).

    Returns:
        List of {section_number, section_title, content, similarity_score},
        sorted by relevance (most relevant first).
    """
    collection = get_collection()

    # Auto-build if empty (first run convenience)
    if collection.count() == 0:
        build_index()

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
    )

    output = []
    for i, doc in enumerate(results['documents'][0]):
        metadata = results['metadatas'][0][i]
        distance = results['distances'][0][i]
        # Chroma returns cosine distance (0 = identical, 2 = opposite)
        # Convert to a friendlier similarity score for logging/debugging
        similarity = round(1 - (distance / 2), 3)

        output.append({
            'section_number': metadata['section_number'],
            'section_title': metadata['section_title'],
            'content': doc,
            'similarity_score': similarity,
        })

    return output


if __name__ == '__main__':
    # Build the index
    print("Building RAG index...\n")
    n = build_index()
    print(f"Indexed {n} sections\n")
    print("=" * 60)

    # Test queries — proves retrieval works for the scenarios you care about
    test_queries = [
        "Can I return fresh salmon?",                                  # perishable
        "What's the refund threshold for automatic approval?",         # $50 rule
        "How many days do I have to return a product?",                # 30-day window
        "My package is lost, what do I do?",                           # lost orders
        "I want to talk to a real human",                              # escalation
        "Can I cancel my order?",                                      # cancellation
        "Are opened cosmetics returnable?",                            # beauty rule
        "My electronics item arrived damaged",                         # defective
    ]

    for q in test_queries:
        results = get_return_policy(q, top_k=2)
        print(f"\nQ: {q}")
        for r in results:
            print(f"  → [{r['similarity_score']}] "
                  f"Section {r['section_number']}: {r['section_title']}")