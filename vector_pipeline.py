import ollama
import chromadb
from chromadb.config import Settings
import uuid

# --- CONFIGURATION ---
MODEL_NAME = "mxbai-embed-large"
DB_PATH = "./my_local_vector_db"
COLLECTION_NAME = "devsecops_docs"


def get_embedding(text):
    """
    Generates a vector embedding for a given text string using Ollama.
    Returns: List[float]
    """
    try:
        # Ollama's embed function returns a dictionary with an 'embeddings' key
        # containing a list of lists (since it can handle batches).
        # We only want the first one for a single string.
        response = ollama.embed(model=MODEL_NAME, input=text)
        return response['embeddings'][0]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []


def main():
    print(f"🚀 Starting Vector Pipeline with model: {MODEL_NAME}...")

    # 1. Initialize ChromaDB (Persistent Client)
    # This saves the database to your disk so data survives after the script ends.
    client = chromadb.PersistentClient(path=DB_PATH)

    # 2. Create or Get Collection
    # We specify metadata usually, but for simplicity, we rely on auto-settings.
    # Note: mxbai-embed-large produces 1024-dimension vectors.
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    print(f"📂 Collection '{COLLECTION_NAME}' ready.")

    # 3. Sample Data (Simulating your DevSecOps corpus)
    documents = [
        "CyberArk Vault is used to secure privileged credentials and rotate passwords automatically.",
        "HashiCorp Vault uses a secret engine to generate dynamic database credentials on the fly.",
        "The 9800X3D is a high-performance CPU optimized for gaming and parallel workloads.",
        "To secure a CI/CD pipeline, ensure no hardcoded secrets exist in the git repository.",
        "Python 3.12 introduced performance improvements and better error messages."
    ]

    # 4. Generate Embeddings & Ingest Data
    print("\n⚡ Generating embeddings and storing in Vector DB...")

    # We prepare lists for batch insertion (faster than one by one)
    ids = []
    embeddings = []
    metadatas = []

    for i, doc in enumerate(documents):
        print(f"   Processing doc {i + 1}/{len(documents)}...")
        vector = get_embedding(doc)

        if vector:
            ids.append(f"doc_{i}")
            embeddings.append(vector)
            metadatas.append({"source": "internal_manual", "doc_id": i})
            # You pass the raw document content to Chroma so it can return it later

    # 5. Add to ChromaDB
    if ids:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,  # Chroma stores the raw text for you automatically
            metadatas=metadatas
        )
        print(f"✅ Successfully stored {len(ids)} documents.")

    # 6. Query the Database
    query_text = "How do I manage secrets in my pipeline?"
    print(f"\n🔍 Querying: '{query_text}'")

    # Generate vector for the query itself
    query_vector = get_embedding(query_text)

    # Search!
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=2  # Get top 2 matches
    )

    # 7. Display Results
    print("\n--- Results ---")
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        score = results['distances'][0][i]  # Lower distance = closer match in Chroma
        print(f"Result {i + 1} (Distance: {score:.4f}):\n   \"{doc}\"\n")


if __name__ == "__main__":
    main()