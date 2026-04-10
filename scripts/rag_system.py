import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add root directory to path to allow importing from src
sys.path.append(os.getcwd())

try:
    from src import (
        Document, 
        RecursiveChunker, 
        EmbeddingStore, 
        LocalEmbedder, 
        KnowledgeBaseAgent
    )
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure you are running the script from the project root.")
    sys.exit(1)

# 1. Initialize Backend
load_dotenv()

def get_rag_components(collection_name: str = "bvi_knowledge_base"):
    """Initialize and return the embedder and store."""
    # This will download the model to your machine on the first run
    embedder = LocalEmbedder() 
    store = EmbeddingStore(collection_name=collection_name, embedding_fn=embedder)
    return embedder, store

def index_data(folder_path: str, collection_name: str = "bvi_knowledge_base"):
    """
    Reads files from a directory, chunks them, and imports them into the vector store.
    """
    _, store = get_rag_components(collection_name)
    path = Path(folder_path)
    
    if not path.exists():
        print(f"Error: Directory {folder_path} not found.")
        return

    chunker = RecursiveChunker(chunk_size=500)
    documents_to_add = []

    print(f"Scanning directory: {folder_path}")
    files = list(path.glob("**/*.md")) + list(path.glob("**/*.txt"))
    
    for file_path in files:
        try:
            content = file_path.read_text(encoding="utf-8")
            chunks = chunker.chunk(content)
            
            for i, chunk_text in enumerate(chunks):
                doc = Document(
                    id=f"{file_path.stem}_{i}",
                    content=chunk_text,
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "doc_id": file_path.stem 
                    }
                )
                documents_to_add.append(doc)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not documents_to_add:
        print("No content found to index.")
        return

    print(f"Found {len(documents_to_add)} chunks. Embedding and importing into Qdrant...")
    store.add_documents(documents_to_add)
    print("Indexing complete!")

def run_rag_query(question: str, collection_name: str = "bvi_knowledge_base"):
    """
    Retrieves context from the DB and generates an answer using the agent.
    """
    _, store = get_rag_components(collection_name)
    
    # Simple Mock LLM for demonstration
    # You can replace this with an actual OpenAI or Ollama call
    def mock_llm(prompt: str) -> str:
        print("\n" + "="*50)
        print("DEBUG: LLM PROMPT SENT")
        print("="*50)
        print(prompt)
        print("="*50 + "\n")
        return "Detailed answer based on the retrieved context above (this is a placeholder)."

    agent = KnowledgeBaseAgent(store=store, llm_fn=mock_llm)
    
    print(f"Querying: '{question}'")
    answer = agent.answer(question, top_k=3)
    return answer

if __name__ == "__main__":
    # Example usage:
    # 1. Index the Vietnamese data we split earlier
    target_dir = "data/ready/vi"
    if os.path.exists(target_dir):
        index_data(target_dir)
    else:
        print(f"Directory {target_dir} not found. Please run scripts/split_bilingual.py first.")

    # 2. Ask a question
    print("\n--- RAG TEST ---")
    query = "Hệ thống R-Evo Smart S có ưu điểm gì về bơm?"
    ans = run_rag_query(query)
    print(f"AGENT RESPONSE: {ans}")
