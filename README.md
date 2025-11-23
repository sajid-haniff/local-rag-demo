# Local RAG Pipeline (Ollama + ChromaDB)

A fully local, private Retrieval-Augmented Generation (RAG) system running on Python. This project demonstrates how to ingest PDF documents, generate embeddings locally, and query them using a Large Language Model without sending data to external APIs.

## 🚀 Features

* **100% Local:** Runs entirely on your machine using [Ollama](https://ollama.com/) and [ChromaDB](https://www.trychroma.com/).
* **Dual Implementation:**
    * `vector_pipeline.py`: A manual, "under-the-hood" implementation to demonstrate core RAG mechanics.
    * `simple_rag.py`: A high-level implementation using **LangChain** for production-ready abstraction.
* **State-of-the-art Models:**
    * Embedding: `mxbai-embed-large` (335M params, 1024 dim).
    * Inference: `llama3.2` (or compatible models like Mistral/Qwen).

## 🛠️ Prerequisites

* **Python 3.10+**
* **Ollama** installed and running.

### 1. Pull Required Models
Before running the code, ensure Ollama has the necessary models:
```bash
# The embedding model (for vectorization)
ollama pull mxbai-embed-large

# The chat model (for answering questions)
ollama pull llama3.2