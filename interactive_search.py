import os
import sys
import logging
from src.retriever.search_logic import RetrievalSystem

logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Welcome to the Glance Fashion Retrieval Engine!")
    print("="*50)
    print("Loading AI models and FAISS indices into GPU memory...")
    
    indices_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'indices'))
    if not os.path.exists(indices_dir):
        print("[ERROR] Indices not found. Please run run_indexer.py first.")
        sys.exit(1)
        
    retriever = RetrievalSystem(indices_dir)
    print("Models loaded successfully!\n")
    
    while True:
        try:
            query = input("\nEnter your search query (or type 'quit' to exit): ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("Exiting search engine. Goodbye!")
                break
                
            if not query:
                continue
                
            print(f"Searching for: '{query}'...")
            results = retriever.search(query, top_k=5, alpha=0.5)
            
            print("\n--- Top Matches ---")
            for i, res in enumerate(results):
                print(f"  {i+1}. {res['file_name']} (Score: {res['score']:.4f})")
            print("-" * 19)
            
        except KeyboardInterrupt:
            print("\nExiting search engine. Goodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] An error occurred: {e}")
