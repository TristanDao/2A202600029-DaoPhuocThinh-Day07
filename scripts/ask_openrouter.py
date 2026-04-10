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
    
    # 2. Step 1: Manual Search to show Top Results first
    top_k = 3
    print(f"[DEBUG] Searching for Top-{top_k} candidates in DB...")
    results = store.search(question, top_k=top_k)
    
    print("\n" + "-"*50)
    print(f"RETRIEVED CONTEXT (TOP {len(results)} RESULTS)")
    print("-"*50)
    if not results:
        print("!!! NO RESULTS FOUND IN DATABASE !!!")
    for i, res in enumerate(results, 1):
        score = res.get('score', 0.0)
        source = res.get('metadata', {}).get('source', 'Unknown')
        content_preview = res.get('content', '')[:300].replace('\n', ' ')
        
        print(f"[{i}] SCORE: {score:.4f}  |  SOURCE: {source}")
        print(f"    CONTENT: {content_preview}...")
        print("-" * 30)
    print("="*50 + "\n")

    # 3. Step 2: Call Agent for Final Answer
    # We pass our OpenRouter call function as the LLM provider
    agent = KnowledgeBaseAgent(store=store, llm_fn=call_openrouter_api)
    
    print("[DEBUG] Generating final answer using OpenRouter...")
    answer = agent.answer(question, top_k=top_k)
    
    print("\n" + "█"*50)
    print("AI RESPONSE")
    print("█"*50)
    print(answer)
    print("█"*50 + "\n")
    
    return answer

if __name__ == "__main__":
    load_dotenv()
    
    # Check if a question was passed as argument for single-shot mode
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_debug_query(query)
    else:
        # INTERACTIVE CHAT MODE
        print("\n" + "╔" + "═"*50 + "╗")
        print("║" + " "*13 + "BVI RAG CHAT SYSTEM ACTIVE" + " "*11 + "║")
        print("╚" + "═"*50 + "╝")
        print("Gõ 'exit', 'quit' hoặc 'q' để thoát chương trình.")
        
        while True:
            try:
                user_input = input("\nBạn muốn hỏi gì?: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("Tạm biệt!")
                    break
                
                if not user_input:
                    continue
                
                run_debug_query(user_input)
            except KeyboardInterrupt:
                print("\nĐã dừng chương trình. Tạm biệt!")
                break
            except Exception as e:
                print(f"Lỗi hệ thống: {e}")
