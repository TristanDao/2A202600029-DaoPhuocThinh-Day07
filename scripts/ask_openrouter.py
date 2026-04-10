import os
import sys
import requests
import json
from pathlib import Path
from dotenv import load_dotenv

# Ensure the script can import the 'src' package
sys.path.append(os.getcwd())

from src import EmbeddingStore, LocalEmbedder, KnowledgeBaseAgent

def call_openrouter_api(prompt: str) -> str:
    """
    Client for OpenRouter API.
    Used as the bridge between our RAG Agent and the LLM.
    """
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
    
    if not api_key or api_key == "your-key-here":
        return "ERROR: OPENROUTER_API_KEY not set in .env file."

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/tristandao", # Optional referer
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
    except Exception as e:
        return f"OpenRouter API Error: {str(e)}"

def run_debug_query(question: str, collection_name: str = "bvi_knowledge_base"):
    """
    Runs a RAG query and prints verbose output for debugging.
    """
    print(f"\n[DEBUG] Question: {question}")
    
    # 1. Setup Backend
    load_dotenv()
    embedder = LocalEmbedder()
    store = EmbeddingStore(collection_name=collection_name, embedding_fn=embedder)
    
    # 2. Setup Agent
    # We pass our OpenRouter call function as the LLM provider
    agent = KnowledgeBaseAgent(store=store, llm_fn=call_openrouter_api)
    
    # 3. Get Answer
    print("[DEBUG] Searching DB and generating answer...")
    answer = agent.answer(question, top_k=3)
    
    print("\n" + "="*50)
    print("FINAL ANSWER")
    print("="*50)
    print(answer)
    print("="*50 + "\n")
    
    return answer

if __name__ == "__main__":
    # Check if a question was passed as argument
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        # Default test question
        query = "Hệ thống R-Evo Smart S có ưu điểm gì nổi bật?"
        
    run_debug_query(query)
