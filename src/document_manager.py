import json
import os
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

class DocumentManager:
    """
    Manages the knowledge base lifecycle: loading, chunking, embedding, and retrieval.
    
    Architecture Notes (for interview):
    ────────────────────────────────────
    CHUNKING STRATEGY:
      We use a "per-article" chunking approach. Each KB article is stored as a single
      Document rather than splitting it into sub-chunks. This is deliberate because:
        1. The KB articles are SHORT (avg ~150 tokens each). Splitting them would
           destroy semantic coherence and hurt retrieval recall.
        2. Keeping articles whole preserves the metadata boundary—every chunk maps
           exactly to one source ID (kb_001, kb_002, etc.), making citation trivial.
        3. If articles were much larger (e.g., 5000+ words), we would switch to
           RecursiveCharacterTextSplitter with overlap to avoid losing context at
           chunk boundaries.
    
    EMBEDDING MODEL CHOICE:
      We chose `all-MiniLM-L6-v2` (384-dim) because:
        1. It runs locally on CPU—no API key, no latency, no cost.
        2. Ranked Top-5 on MTEB for its size class in English similarity tasks.
        3. For a larger-scale production system, we would upgrade to:
           - `all-mpnet-base-v2` (768-dim, +3% accuracy, 2x slower)
           - Or API-based embeddings like `text-embedding-3-large` for multilingual.
    
    VECTOR STORE:
      ChromaDB is used for zero-config local persistence. In production, we would
      migrate to Pinecone, Qdrant, or Postgres+pgvector for horizontal scaling.
    """
    
    def __init__(self, data_path: str = "data/knowledge_base.json", persist_directory: str = "chroma_db"):
        self.data_path = data_path
        self.persist_directory = persist_directory
        # Force CPU device to avoid macOS MPS graph permission errors in sandboxed environments
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        self.vector_store = None

    def load_documents(self) -> List[Document]:
        """
        Load the JSON knowledge base into LangChain Documents.
        
        Each article becomes ONE document. We attach rich metadata so the
        retriever and LLM can reason about status, recency, and category.
        """
        with open(self.data_path, "r") as f:
            data = json.load(f)
        
        documents = []
        for item in data:
            metadata = {
                "id": item["id"],
                "title": item["title"],
                "category": item["category"],
                "status": item["status"],
                "last_updated": item["last_updated"],
            }
            # tags might be a list, Chroma requires strings/ints/floats
            if "tags" in item:
                metadata["tags"] = ", ".join(item["tags"])
            
            doc = Document(page_content=item["content"], metadata=metadata)
            documents.append(doc)
        
        return documents

    def initialize_vector_store(self, force_reload: bool = False) -> Chroma:
        """
        Create or load the ChromaDB vector store.
        
        Uses persist_directory so embeddings are computed only once.
        Subsequent loads skip re-embedding entirely (fast cold start).
        """
        if os.path.exists(self.persist_directory) and not force_reload:
            print("Loading existing vector store...")
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            print("Creating new vector store...")
            documents = self.load_documents()
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        
        return self.vector_store

if __name__ == "__main__":
    manager = DocumentManager()
    manager.initialize_vector_store(force_reload=True)
    print("Vector store initialized successfully.")
