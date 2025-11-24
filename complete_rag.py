import os
import sys
import time
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
# Hardware / Model Settings
EMBED_MODEL = "mxbai-embed-large"  # 335M params, 1024 dims
CHAT_MODEL = "llama3.2"  # 3B params (Fast & Efficient)

# Data Settings
PDF_PATH = "./pension.pdf"
DB_PATH = "./local_rag_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def ingest_document():
    """
    Loads the PDF, splits it into chunks, and embeds them into ChromaDB.
    Only runs if the DB is empty or missing.
    """
    # 1. Initialize Embedding Function (The "Translator")
    embedding_function = OllamaEmbeddings(model=EMBED_MODEL)

    # 2. Check if DB exists and has data
    if os.path.exists(DB_PATH):
        # We try to connect to see if it's actually populated
        test_client = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
        # Chroma's get() returns a dict with 'ids', 'embeddings', etc.
        if len(test_client.get()['ids']) > 0:
            print(f"✅ Database found at {DB_PATH}. Skipping ingestion.")
            return test_client

    print(f"📂 Loading {PDF_PATH}...")
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: File {PDF_PATH} not found.")
        sys.exit(1)

    # 3. Load (PyMuPDF is fast and handles formatting well)
    loader = PyMuPDFLoader(PDF_PATH)
    raw_documents = loader.load()
    print(f"   Loaded {len(raw_documents)} pages.")

    # 4. Split (Recursive logic preserves paragraph structure)
    print("✂️  Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"   Created {len(chunks)} semantic chunks.")

    # 5. Index (Embed + Store)
    print("💾 Vectorizing and Storing (This may take a moment)...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=DB_PATH
    )
    print("✅ Ingestion Complete.")
    return vector_db


def get_rag_chain(vector_db):
    """
    Builds the RAG pipeline: Retriever -> Prompt -> LLM -> Parser
    """
    # 1. Retriever (The Search Engine)
    # k=5 retrieves top 5 chunks. 'mmr' (Maximal Marginal Relevance) ensures diversity.
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )

    # 2. The Chat Model (The Brain)
    llm = ChatOllama(model=CHAT_MODEL, temperature=0.1)  # Low temp for factual answers

    # 3. The Prompt (The Instructions)
    template = """You are a specialized assistant for analyzing technical documents.
    Answer the question based ONLY on the following context. 
    If the answer is not in the context, strictly state "I don't know based on the provided text."

    --- CONTEXT ---
    {context}

    --- QUESTION ---
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 4. The Chain (The Assembly Line)
    # RunnablePassthrough passes the user's question to the prompt
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return chain


def main():
    print("🚀 Starting Local RAG Pipeline...")

    # Step 1: Ensure Data is Ready
    vector_db = ingest_document()

    # Step 2: Build the Chain
    rag_chain = get_rag_chain(vector_db)

    print(f"\n💬 RAG System Online ({CHAT_MODEL} + {EMBED_MODEL})")
    print("Type 'exit' to quit.\n")

    # Step 3: Chat Loop
    while True:
        try:
            query = input("👉 Question: ")
            if query.lower() in ['exit', 'quit', 'q']:
                break

            print("🤖 Answer: ", end="", flush=True)

            # Stream the response for a better UI experience
            start_time = time.time()
            for chunk in rag_chain.stream(query):
                print(chunk, end="", flush=True)

            end_time = time.time()
            print(f"\n   (⏱️  {end_time - start_time:.2f}s)\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()