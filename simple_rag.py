import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# --- CONFIGURATION ---
PDF_PATH = "./pension.pdf"  # Put your PDF here
DB_PATH = "./simple_local_db"
MODEL_NAME = "mxbai-embed-large"


def main():
    # 1. LOAD (The Simple Way)
    # PyMuPDF is 10x faster than PyPDF2 and handles formatting better
    if not os.path.exists(PDF_PATH):
        print(f"❌ File not found: {PDF_PATH}")
        return

    print(f"📂 Loading {PDF_PATH}...")
    loader = PyMuPDFLoader(PDF_PATH)
    raw_documents = loader.load()
    print(f"   Loaded {len(raw_documents)} pages.")

    # 2. SPLIT (The Standard Way)
    # recursive splitting tries to keep paragraphs together.
    # chunk_size=1000: A good balance for retrieval.
    # chunk_overlap=200: CRITICAL. Ensures context isn't cut off at the edges.
    print("✂️  Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]  # Try to split by paragraph first
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"   Created {len(chunks)} chunks.")

    # 3. EMBED & STORE
    print("💾 Creating Vector Database (this relies on Ollama)...")
    # This will pull the model if you haven't already
    embedding_function = OllamaEmbeddings(model=MODEL_NAME)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=DB_PATH,
        collection_name="simple_pdf_collection"
    )
    print("✅ Database ready.")

    # 4. QUERY
    query_text = "How does your pension grow?"
    print(f"\n🔎 Querying: '{query_text}'")

    # Retrieve top 3 matches
    results = vector_db.similarity_search_with_score(query_text, k=3)

    # 5. DISPLAY RESULTS
    print("\n--- Results ---")
    for doc, score in results:
        # Chroma returns distance (lower is better)
        print(f"SCORE: {score:.4f}")
        print(f"PAGE: {doc.metadata.get('page', 'N/A')}")
        print(f"CONTENT: {doc.page_content[:300]}...\n")


if __name__ == "__main__":
    main()