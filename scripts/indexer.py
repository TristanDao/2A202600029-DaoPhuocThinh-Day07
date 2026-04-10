import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure the script can import the 'src' package
sys.path.append(os.getcwd())

from src import Document, RecursiveChunker, EmbeddingStore, LocalEmbedder

def index_data_idempotent(folder_path: str, collection_name: str = "bvi_knowledge_base"):
    """
    Scans a folder for documents and imports them into the vector store.
    Checks if a file (based on its stem) has already been indexed to avoid duplication.
    """
    load_dotenv()
    
    # Initialize the local embedding model (downloaded to the machine)
    embedder = LocalEmbedder()
    store = EmbeddingStore(collection_name=collection_name, embedding_fn=embedder)
    
    # Intelligent recursive chunker
    chunker = RecursiveChunker(chunk_size=600)

    path = Path(folder_path)
    if not path.exists():
        print(f"Error: Path {folder_path} does not exist.")
        return

    # Find all Markdown and Text files
    files = list(path.glob("**/*.md")) + list(path.glob("**/*.txt"))
    
    print(f"Analyzing {len(files)} files in '{folder_path}'...")

    new_docs_count = 0
    skipped_count = 0

    for file_path in files:
        # Use filename without extension as the unique doc_id
        doc_id = file_path.stem
        
        # IDEMPOTENCY CHECK:
        # We query the store to see if any chunks with this 'doc_id' metadata already exist.
        # We use a dummy query 'test' since we only care about the metadata filter.
        existing_chunks = store.search_with_filter(
            query="check existence", 
            top_k=1, 
            metadata_filter={"doc_id": doc_id}
        )
        
        if existing_chunks:
            print(f"[-] SKIPPED: '{doc_id}' already indexed.")
            skipped_count += 1
            continue

        print(f"[+] INDEXING: '{doc_id}'...")
        try:
            content = file_path.read_text(encoding="utf-8")
            text_chunks = chunker.chunk(content)
            
            docs_to_store = []
            for i, chunk_text in enumerate(text_chunks):
                doc = Document(
                    id=f"{doc_id}_{i}",
                    content=chunk_text,
                    metadata={
                        "source": str(file_path),
                        "doc_id": doc_id,
                        "chunk_index": i
                    }
                )
                docs_to_store.append(doc)
            
            store.add_documents(docs_to_store)
            new_docs_count += 1
        except Exception as e:
            print(f"Error processing {doc_id}: {e}")

    print("\n" + "="*30)
    print(f"Indexing Summary:")
    print(f" - Total files processed: {len(files)}")
    print(f" - New files added: {new_docs_count}")
    print(f" - Files skipped (already existed): {skipped_count}")
    print("="*30)

if __name__ == "__main__":
    # Point to the Vietnamese folder created by the split script
    target_folder = "data/ready/vi"
    index_data_idempotent(target_folder)
